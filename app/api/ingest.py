"""
Upload a PDF, chunk it, merge into FAISS.

This is the first half of RAG. The second half (retriever tool on the agent)
comes once you have a few Bronner precedents indexed.
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.rag.ingest_pdf import ingest_pdf
from app.rag.store import build_or_merge_store

router = APIRouter(prefix="/ingest", tags=["ingest"])


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

    dest = settings.uploads_dir / Path(file.filename).name
    content = await file.read()
    dest.write_bytes(content)

    chunks = ingest_pdf(dest, source_name=dest.name)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text in this PDF.")

    build_or_merge_store(chunks)
    return {
        "filename": dest.name,
        "chunks": len(chunks),
        "message": "Indexed into FAISS. Agent retrieval tool comes next.",
    }
