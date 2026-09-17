"""
FAISS vector store helpers.

Same building blocks as your LangChain RAG.py: embed texts → FAISS →
as_retriever. Persistence path comes from Settings so uploads survive restarts.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.llm.openrouter import get_embeddings
from app.rag.chunking import TextChunk
from app.storage.blob_index import hydrate_faiss_index, persist_faiss_index


def chunks_to_documents(chunks: list[TextChunk]) -> list[Document]:
    docs = []
    for c in chunks:
        docs.append(
            Document(
                page_content=c.text,
                metadata={"source": c.source, "page": c.page, "index": c.index},
            )
        )
    return docs


def ensure_index_ready() -> None:
    settings = get_settings()
    hydrate_faiss_index(settings.faiss_dir, settings.bundled_faiss_dir)


def build_or_merge_store(chunks: list[TextChunk]) -> FAISS:
    """Create a new index or merge into the existing one on disk."""
    ensure_index_ready()
    settings = get_settings()
    embeddings = get_embeddings()
    docs = chunks_to_documents(chunks)
    index_path = settings.faiss_dir

    if _index_exists(index_path):
        store = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        store.add_documents(docs)
    else:
        store = FAISS.from_documents(docs, embedding=embeddings)

    store.save_local(str(index_path))
    persist_faiss_index(index_path)
    return store


def load_store() -> FAISS | None:
    ensure_index_ready()
    settings = get_settings()
    if not _index_exists(settings.faiss_dir):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(str(settings.faiss_dir), embeddings, allow_dangerous_deserialization=True)


def _index_exists(path: Path) -> bool:
    return (path / "index.faiss").exists()


def _all_documents(store: FAISS) -> list[Document]:
    docs: list[Document] = []
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        if doc is not None:
            docs.append(doc)
    return docs


def _source_kind(source: str) -> str:
    return "bootstrap" if source.endswith("(Oyez summary)") else "upload"


def list_index_sources() -> list[dict]:
    """Unique source strings in the FAISS docstore with chunk counts."""
    store = load_store()
    if store is None:
        return []

    counts: Counter[str] = Counter()
    for doc in _all_documents(store):
        source = str((doc.metadata or {}).get("source") or "unknown")
        counts[source] += 1

    rows = []
    for source, chunk_count in sorted(counts.items(), key=lambda item: item[0].lower()):
        rows.append(
            {
                "source": source,
                "chunks": chunk_count,
                "kind": _source_kind(source),
            }
        )
    return rows


def _source_matches(metadata_source: str, target: str) -> bool:
    if metadata_source == target:
        return True
    # Allow DELETE by bare filename when metadata stores the upload name.
    if metadata_source.endswith(target) and target.lower().endswith(".pdf"):
        return True
    return False


def remove_source_from_index(source: str) -> int:
    """
    Drop every chunk whose metadata source matches `source`.
    Rebuilds FAISS from remaining documents and persists to disk/Blob.
    Returns number of chunks removed.
    """
    store = load_store()
    if store is None:
        return 0

    docs = _all_documents(store)
    kept: list[Document] = []
    removed = 0
    for doc in docs:
        metadata_source = str((doc.metadata or {}).get("source") or "")
        if _source_matches(metadata_source, source):
            removed += 1
        else:
            kept.append(doc)

    if removed == 0:
        return 0

    settings = get_settings()
    index_path = settings.faiss_dir
    embeddings = get_embeddings()

    if kept:
        new_store = FAISS.from_documents(kept, embedding=embeddings)
        new_store.save_local(str(index_path))
        persist_faiss_index(index_path)
    else:
        _clear_index(index_path)

    return removed


def _clear_index(index_path: Path) -> None:
    if index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    persist_faiss_index(index_path)


def delete_uploaded_pdf(filename: str) -> None:
    """Remove saved PDF bytes from the writable uploads folder, if present."""
    settings = get_settings()
    path = settings.uploads_dir / Path(filename).name
    if path.is_file():
        path.unlink()
