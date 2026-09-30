"""
Durable copies of Ask AI corpus PDFs in Vercel Blob.

Local uploads/ lives under /tmp on Vercel, so a cold start or redeploy wipes
those bytes even when FAISS still knows the source name. Keep a Blob mirror
keyed by filename so /ingest/file can rehydrate after the disk is gone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

from app.storage.blob_client import blob_configured, delete_blob, get_blob, put_blob

logger = logging.getLogger(__name__)

_INGEST_FILES_PREFIX = "case-law-agent/ingest-files"


def ingest_file_blob_pathname(filename: str) -> str:
    """Stable Blob path for a corpus PDF, keyed by the display filename."""
    safe = Path(filename).name
    return f"{_INGEST_FILES_PREFIX}/{quote(safe, safe='')}"


def persist_uploaded_pdf(filename: str, content: bytes) -> None:
    """
    Mirror PDF bytes to Blob when credentials exist.

    Local save still happens first (see _index_pdf_bytes). This call keeps a
    durable copy for the next serverless instance.
    """
    if not blob_configured() or not content:
        return
    pathname = ingest_file_blob_pathname(filename)
    put_blob(pathname, content, content_type="application/pdf")
    logger.info("Persisted ingest PDF to Blob pathname=%s bytes=%s", pathname, len(content))


def load_uploaded_pdf(filename: str) -> bytes | None:
    """Return PDF bytes from Blob, or None when missing / Blob is off."""
    if not blob_configured():
        return None
    try:
        return get_blob(ingest_file_blob_pathname(filename))
    except Exception:
        logger.exception("Could not load ingest PDF %s from Blob", filename)
        return None


def delete_uploaded_pdf_blob(filename: str) -> None:
    """Drop the Blob mirror when a source is removed from Ask AI."""
    if not blob_configured():
        return
    delete_blob(ingest_file_blob_pathname(filename))
