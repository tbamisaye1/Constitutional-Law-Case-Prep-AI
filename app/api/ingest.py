"""
Upload a PDF, chunk it, merge into FAISS.

This is the first half of RAG. The second half (retriever tool on the agent)
comes once you have a few Bronner precedents indexed.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.rag.ingest_pdf import ingest_pdf
from app.rag.store import build_or_merge_store

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _writable_data_dir() -> Path | None:
    """Return a directory we can write PDFs and FAISS into, or None on read-only hosts."""
    if os.environ.get("VERCEL"):
        return None
    settings = get_settings()
    for candidate in (settings.data_dir, Path(tempfile.gettempdir()) / "case-law-agent-data"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return None


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

    if _writable_data_dir() is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "PDF upload is not available on the hosted demo (read-only server). "
                "Run locally: uvicorn app.main:app --port 8000 and npm run dev, "
                "then open http://localhost:5173/upload."
            ),
        )

    dest = settings.uploads_dir / Path(file.filename).name
    content = await file.read()
    try:
        dest.write_bytes(content)
    except OSError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot save upload on this host ({e}). Use local dev for PDF ingest.",
        ) from e

    chunks = ingest_pdf(dest, source_name=dest.name)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text in this PDF.")

    try:
        build_or_merge_store(chunks)
    except OSError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot update search index on this host ({e}). Use local dev for PDF ingest.",
        ) from e

    return {
        "filename": dest.name,
        "chunks": len(chunks),
        "message": "Indexed into FAISS. Agent retrieval tool comes next.",
    }
