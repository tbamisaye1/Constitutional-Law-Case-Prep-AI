"""
Split PDF text into chunks for embedding.

Design choice: simple character windows first. Once we care about
holding-vs-dicta quality, we can switch to recursive / section-aware
chunking without changing the store API.
"""

from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    source: str
    page: int | None = None
    index: int = 0


def chunk_text(text: str, source: str, page: int | None = None, size: int = 1200, overlap: int = 200) -> list[TextChunk]:
    """Walk the string with overlap so cites near boundaries are not lost."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[TextChunk] = []
    start = 0
    i = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        piece = cleaned[start:end]
        chunks.append(TextChunk(text=piece, source=source, page=page, index=i))
        i += 1
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks
