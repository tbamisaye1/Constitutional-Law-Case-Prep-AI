"""
PDF bytes for the case library.

Two Vercel limits shape this module, and both are hard infrastructure caps that
no vercel.json setting can raise:

* A request body may not exceed 4.5 MB, so uploads are capped just under that
  and anything larger gets a 413 that explains itself.
* A response body may not exceed 4.5 MB either, so downloads never stream
  through Python. GET redirects to Blob's CDN instead, which also keeps the
  function's execution time out of the picture for a large record.

Metadata still travels over /sync with the rest of the store. This module only
owns the bytes and the blob pointer, which is why the sync entity spec has no
blobPathname field: a client cannot aim a document row at someone else's blob.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from app.api.deps import WorkspaceSession, workspace_session
from app.db.repository import (
    document_to_wire,
    get_document,
    soft_delete_document,
    upsert_document_blob,
)
from app.storage.blob_client import blob_configured, delete_blob, put_blob

router = APIRouter(prefix="/documents", tags=["documents"])

# Vercel rejects request bodies over 4.5 MB before our code runs. Sitting at
# 4 MB leaves room for the multipart envelope and form fields, so the caller
# gets our readable error instead of a bare platform 413.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

_BLOB_PREFIX = "case-law-agent/workspaces"


def _blob_pathname(workspace_id: str, document_id: str) -> str:
    """
    Where one document's bytes live.

    The random segment is what makes the resulting public URL unguessable.
    Blob objects here are readable by anyone holding the URL, so the secret has
    to be in the path itself. The document id alone would not do: it is minted
    client-side from a timestamp.
    """
    safe_id = secrets.token_urlsafe(12)
    return f"{_BLOB_PREFIX}/{workspace_id}/{document_id}/{safe_id}.pdf"


@router.post("")
async def upload_document(
    file: UploadFile = File(..., description="The PDF bytes."),
    document_id: str = Form(..., description="Client-generated id, matching filesMeta."),
    case_id: str = Form(..., description="Library case id, or 'case-at-bar'."),
    session: WorkspaceSession = Depends(workspace_session),
) -> dict:
    """
    Store a PDF in Blob and record it against this workspace.

    The client has already written the file to IndexedDB by the time it calls
    this, so a failure here degrades to browser-only storage rather than losing
    the upload.

    Raises:
        HTTPException: 400 for a non-PDF or a blank id, 413 when the file is
            over MAX_UPLOAD_BYTES, 503 when Blob is not configured.
    """
    if not document_id.strip() or not case_id.strip():
        raise HTTPException(status_code=400, detail="document_id and case_id are required.")

    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file.")

    if not blob_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Blob storage is not configured on this host. "
                "Set BLOB_READ_WRITE_TOKEN, or keep working with browser-only storage."
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{filename} is {len(content) / 1024 / 1024:.1f} MB. Uploads through "
                f"this API are capped at {MAX_UPLOAD_BYTES // (1024 * 1024)} MB by "
                "Vercel's request body limit. Split the record, or keep this file "
                "in browser-only storage for now."
            ),
        )

    pathname = _blob_pathname(session.workspace_id, document_id.strip())
    try:
        stored = put_blob(pathname, content, content_type="application/pdf")
    except (httpx.HTTPError, RuntimeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Blob rejected the upload: {error}",
        ) from error

    now = datetime.now(timezone.utc)
    return upsert_document_blob(
        session.cursor,
        workspace_id=session.workspace_id,
        document_id=document_id.strip(),
        case_id=case_id.strip(),
        name=filename,
        size_bytes=len(content),
        content_type="application/pdf",
        blob_pathname=pathname,
        blob_url=stored.get("url") or "",
        now=now,
    )


@router.get("/{document_id}")
def read_document(
    document_id: str,
    session: WorkspaceSession = Depends(workspace_session),
) -> dict:
    """Metadata for one document, so a client can tell whether bytes exist."""
    row = get_document(session.cursor, session.workspace_id, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such document in this workspace.")
    return document_to_wire(row)


@router.get("/{document_id}/file")
def download_document(
    document_id: str,
    session: WorkspaceSession = Depends(workspace_session),
) -> RedirectResponse:
    """
    Redirect to the PDF in Blob storage.

    A redirect rather than a proxied stream, because a proxied response would
    hit Vercel's 4.5 MB response cap and would bill function time for bytes the
    CDN serves better. The trade-off is that the client follows a URL which is
    unguessable but not access-controlled.

    Raises:
        HTTPException: 404 when the document is unknown, 409 when the row
            exists but its bytes were never uploaded.
    """
    row = get_document(session.cursor, session.workspace_id, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such document in this workspace.")

    url = row["blob_url"]
    if not url:
        raise HTTPException(
            status_code=409,
            detail=(
                "This document is recorded but its bytes were never uploaded. "
                "Re-add the PDF from the device that has it."
            ),
        )
    # 307 keeps the method and tells caches nothing is permanent here; the blob
    # pathname changes on every re-upload.
    return RedirectResponse(url=url, status_code=307)


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    session: WorkspaceSession = Depends(workspace_session),
) -> dict:
    """
    Tombstone a document and remove its bytes.

    The row is kept as a tombstone so other devices learn about the delete on
    their next sync instead of pushing the document back.
    """
    now = datetime.now(timezone.utc)
    pathname = soft_delete_document(session.cursor, session.workspace_id, document_id, now)

    blob_removed = False
    if pathname:
        try:
            delete_blob(pathname)
            blob_removed = True
        except (httpx.HTTPError, RuntimeError):
            # The tombstone is the part that matters for sync. A leftover blob
            # is wasted storage, not a correctness problem, so do not fail the
            # request over it.
            blob_removed = False

    return {"id": document_id, "deleted": True, "blobRemoved": blob_removed}
