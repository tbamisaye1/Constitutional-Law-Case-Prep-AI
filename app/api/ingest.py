"""
Upload a PDF, chunk it, merge into FAISS.

This is the first half of RAG. The second half (retriever tool on the agent)
comes once you have a few Bronner precedents indexed.
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.ingest_pdf import ingest_pdf
from app.rag.store import (
    build_or_merge_store,
    delete_uploaded_pdf,
    list_index_sources,
    remove_source_from_index,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class RemoveSourceRequest(BaseModel):
    source: str = Field(min_length=1)


@router.get("/sources")
def list_sources_route():
    return {"sources": list_index_sources()}


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

    try:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot prepare upload directory on this host ({e}).",
        ) from e

    dest = settings.uploads_dir / Path(file.filename).name
    content = await file.read()
    try:
        dest.write_bytes(content)
    except OSError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot save upload on this host ({e}).",
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
