"""
Turn an uploaded PDF into TextChunks ready for FAISS.

pypdf keeps the dependency light. We pass page numbers through so the
agent can cite 'source p.12' the way you cite R. pages in the guide.
"""

from pathlib import Path

from pypdf import PdfReader

from app.rag.chunking import TextChunk, chunk_text


def ingest_pdf(path: Path, source_name: str | None = None) -> list[TextChunk]:
    reader = PdfReader(str(path))
    source = source_name or path.name
    all_chunks: list[TextChunk] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        all_chunks.extend(chunk_text(text, source=source, page=page_num))

    return all_chunks
