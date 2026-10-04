"""
Upload a PDF, chunk it, merge into FAISS.

Two ingest paths exist because Vercel caps API request bodies at 4.5 MB:

* POST /ingest/pdf — multipart through the function (small PDFs)
* POST /ingest/blob-client-upload + browser PUT + POST /ingest/from-blob —
  large opinions skip the body cap the same way /documents does

The second half of RAG (retriever tool on the agent) comes once you have a few
Bronner precedents indexed.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.ingest_pdf import ingest_pdf
from app.rag.store import (
    build_or_merge_store,
    delete_uploaded_pdf,
    list_index_sources,
    remove_source_from_index,
)
from app.storage.blob_client import (
    blob_configured,
    generate_client_token,
    read_write_token_required,
    store_slug,
)
from app.storage.ingest_files import load_uploaded_pdf, persist_uploaded_pdf

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Leave headroom under Vercel's 4.5 MB body cap for multipart overhead.
MAX_PROXY_UPLOAD_BYTES = 4 * 1024 * 1024
# Direct browser → Blob uploads can be larger; 50 MB covers full opinions.
MAX_DIRECT_UPLOAD_BYTES = 50 * 1024 * 1024
_INGEST_BLOB_PREFIX = "case-law-agent/ingest"


class RemoveSourceRequest(BaseModel):
    source: str = Field(min_length=1)


class IngestFromBlobRequest(BaseModel):
    filename: str = Field(min_length=1)
    blob_url: str = Field(min_length=1)
    blob_pathname: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)


@router.get("/sources")
def list_sources_route():
    return {"sources": list_index_sources()}


@router.get("/file/{filename}")
def get_uploaded_pdf_route(filename: str):
    """
    Serve a PDF that was saved during ingest.

    Prefer the local uploads/ copy (fast on a warm instance). On Vercel that
    folder is under /tmp, so after a cold start we rehydrate from the durable
    Blob mirror and cache it locally for the rest of the instance life.
    """
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename.replace("\\", "/").split("/")[-1]:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are served.")

    settings = get_settings()
    path = settings.uploads_dir / safe_name
    if path.is_file():
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=safe_name,
            content_disposition_type="inline",
        )

    remote = load_uploaded_pdf(safe_name)
    if remote:
        try:
            settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(remote)
        except OSError:
            # Still serve from memory if /tmp is full or read-only.
            return Response(
                content=remote,
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
            )
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=safe_name,
            content_disposition_type="inline",
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"No uploaded PDF named '{safe_name}'. "
            "Re-upload via /ingest/pdf or attach it in Case library."
        ),
    )


@router.delete("/source")
def remove_source_route(body: RemoveSourceRequest):
    removed = remove_source_from_index(body.source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No indexed chunks matched source '{body.source}'.")

    if body.source.lower().endswith(".pdf"):
        delete_uploaded_pdf(body.source)

    return {
        "source": body.source,
        "chunks_removed": removed,
        "message": "Removed from search index.",
    }


@router.post("/pdf")
async def ingest_pdf_route(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file.")

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY missing; embeddings need it.",
        )

    content = await file.read()
    if len(content) > MAX_PROXY_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"This PDF is {len(content) / (1024 * 1024):.1f} MB. "
                "Vercel caps API uploads at 4.5 MB. The app should retry via "
                "direct Blob ingest; if you still see this, re-upload after refreshing."
            ),
        )

    return _index_pdf_bytes(content, Path(file.filename).name)


@router.post("/blob-client-upload")
async def ingest_blob_client_upload(request: Request) -> dict:
    """
    Mint a short-lived Blob client token for a browser PUT of a large PDF.

    Same protocol as /documents/blob-client-upload, but the pathname sits under
    the shared ingest prefix rather than a workspace document folder.
    """
    try:
        read_write_token_required()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        body = await request.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail="Expected JSON body.") from error

    event_type = body.get("type")
    if event_type == "blob.upload-completed":
        return {"type": "blob.upload-completed", "response": "ok"}

    if event_type != "blob.generate-client-token":
        raise HTTPException(status_code=400, detail=f"Unsupported upload event: {event_type}")

    pathname = f"{_INGEST_BLOB_PREFIX}/{secrets.token_urlsafe(12)}.pdf"
    payload = body.get("payload") or {}
    client_payload = payload.get("clientPayload") or ""

    try:
        client_token = generate_client_token(
            pathname,
            # Match /documents: some browsers label PDFs as octet-stream.
            allowed_content_types=["application/pdf", "application/octet-stream"],
            maximum_size_in_bytes=MAX_DIRECT_UPLOAD_BYTES,
            add_random_suffix=False,
            allow_overwrite=False,
            token_payload=client_payload or None,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "type": "blob.generate-client-token",
        "clientToken": client_token,
        "pathname": pathname,
    }


@router.post("/from-blob")
async def ingest_from_blob_route(body: IngestFromBlobRequest):
    """
    Download a PDF the browser already PUT to Blob, then index it into FAISS.

    The request body stays tiny (URL + metadata). Bytes never enter the API
    function through the request body, so files over 4.5 MB work on Vercel.
    """
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY missing; embeddings need it.",
        )
    if not blob_configured():
        raise HTTPException(status_code=503, detail="Blob storage is not configured.")

    filename = Path(body.filename).name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a .pdf file.")
    if body.size_bytes > MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ingest uploads are capped at {MAX_DIRECT_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    pathname = (body.blob_pathname or "").strip().lstrip("/")
    if not pathname.startswith(f"{_INGEST_BLOB_PREFIX}/") or ".." in pathname:
        raise HTTPException(status_code=400, detail="Blob pathname is not an ingest upload.")
    if not _is_allowed_blob_url(body.blob_url, pathname):
        raise HTTPException(status_code=400, detail="Blob URL is not from this store.")

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(body.blob_url.strip())
            response.raise_for_status()
            content = response.content
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not download PDF from Blob: {error}",
        ) from error

    if not content:
        raise HTTPException(status_code=422, detail="Downloaded PDF was empty.")
    if len(content) > MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ingest uploads are capped at {MAX_DIRECT_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    return _index_pdf_bytes(content, filename)


def _is_allowed_blob_url(url: str, pathname: str) -> bool:
    """Reject non-Blob hosts so /from-blob cannot be used as an open proxy."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False
    if parsed.scheme not in ("https", "http"):
        return False
    host = (parsed.hostname or "").lower()
    if host == "blob.vercel-storage.com":
        return True
    slug = store_slug()
    if slug and host == f"{slug}.public.blob.vercel-storage.com":
        return True
    if host.endswith(".public.blob.vercel-storage.com") and pathname in (parsed.path or ""):
        return True
    return False


def _index_pdf_bytes(content: bytes, filename: str) -> dict:
    """Write bytes under uploads/, mirror to Blob, chunk, merge into FAISS."""
    settings = get_settings()
    try:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot prepare upload directory on this host ({e}).",
        ) from e

    dest = settings.uploads_dir / filename
    try:
        dest.write_bytes(content)
    except OSError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot save upload on this host ({e}).",
        ) from e

    try:
        persist_uploaded_pdf(filename, content)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Indexed PDF could not be persisted to Blob: {e}",
        ) from e

    chunks = ingest_pdf(dest, source_name=dest.name)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text in this PDF.")

    try:
        build_or_merge_store(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update search index: {e}",
        ) from e

    return {
        "filename": dest.name,
        "chunks": len(chunks),
        "message": "Indexed into FAISS. Ask AI can search this case now.",
    }
