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
    if source.endswith("(Oyez summary)"):
        return "bootstrap"
    if is_instant_case_source(source):
        return "instant_case"
    return "upload"


def is_instant_case_source(source: str) -> bool:
    """True for Instant Case / Bronner record PDFs in the FAISS index."""
    name = (source or "").strip().lower()
    if not name or name.endswith("(oyez summary)"):
        return False
    if "instant case" in name or "joint appendix" in name:
        return True
    if "bronner" in name and "hamdi" not in name:
        return True
    # Classroom / Drive exports of the record often land as ACFrOg….pdf
    base = Path(name).name
    if base.startswith("acfrog"):
        return True
    return False


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
    if not target:
        return False
    if metadata_source == target:
        return True
    # Allow DELETE by bare filename when metadata stores the upload name.
    if metadata_source.endswith(target) and target.lower().endswith(".pdf"):
        return True
    # Match basename either way (uploads sometimes store a path prefix).
    meta_base = Path(metadata_source).name.lower()
    target_base = Path(target).name.lower()
    if meta_base and target_base and meta_base == target_base:
        return True
    # Instant Case label: "Instant Case [Bronner v. USA] — file.pdf"
    if " — " in metadata_source:
        suffix = metadata_source.split(" — ", 1)[-1].strip().lower()
        if suffix and (suffix == target_base or suffix == target.lower()):
            return True
    return False


def docs_for_source(
    store: FAISS,
    source: str,
    *,
    page: int | None = None,
    page_window: int = 1,
    limit: int = 6,
) -> list[tuple[Document, float]]:
    """
    Pull chunks from one uploaded PDF, optionally near a page.

    Used when Ask AI opens from a highlight so the model sees the same article
    the student is reading, not a random Bronner / Oyez neighbor.
    """
    if store is None or not (source or "").strip():
        return []

    scored: list[tuple[Document, float]] = []
    for doc in _all_documents(store):
        meta = doc.metadata or {}
        metadata_source = str(meta.get("source") or "")
        if not _source_matches(metadata_source, source):
            continue
        doc_page = meta.get("page")
        distance = 0.05
        if page is not None and doc_page is not None:
            try:
                delta = abs(int(doc_page) - int(page))
            except (TypeError, ValueError):
                delta = 0
            if delta > page_window:
                continue
            # Same page first, then neighbors.
            distance = 0.01 + (0.02 * delta)
        scored.append((doc, distance))

    scored.sort(key=lambda item: (item[1], str((item[0].metadata or {}).get("page") or "")))
    return scored[:limit]


def docs_for_instant_case(
    store: FAISS,
    query: str = "",
    *,
    limit: int = 8,
) -> list[tuple[Document, float]]:
    """
    Pull Instant Case / Bronner record chunks ranked by the question.

    Do not prefer early PDF pages — that used to return only the caption while
    Instant Case questions about appellate reasoning live later in the record.
    """
    if store is None:
        return []

    q = (query or "").strip()
    if q:
        # Over-fetch, then keep Instant Case hits so Hamdi/etc. cannot crowd out
        # the record when the question is about Bronner.
        pairs = store.similarity_search_with_score(q, k=max(48, limit * 6))
        filtered: list[tuple[Document, float]] = []
        seen: set[str] = set()
        for doc, score in pairs:
            meta = doc.metadata or {}
            source = str(meta.get("source") or "")
            if not is_instant_case_source(source):
                continue
            text = (doc.page_content or "").strip()
            key = f"{source}|{meta.get('page')}|{text[:120]}"
            if not text or key in seen:
                continue
            seen.add(key)
            filtered.append((doc, float(score) if score is not None else 1.0))
        if filtered:
            # FAISS L2: lower distance is better. Re-sort after filtering.
            filtered.sort(key=lambda item: item[1])
            return filtered[:limit]

    # Keyword fallback when vector search returns no Instant Case rows.
    import re

    tokens = {t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 3}
    scored: list[tuple[Document, float]] = []
    for doc in _all_documents(store):
        meta = doc.metadata or {}
        source = str(meta.get("source") or "")
        if not is_instant_case_source(source):
            continue
        text = (doc.page_content or "").strip()
        if not text:
            continue
        lower = text.lower()
        hits = sum(1 for t in tokens if t in lower) if tokens else 0
        # Lower is better (same convention as FAISS L2 distance).
        distance = 1.0 / (1.0 + hits) if tokens else 0.5
        scored.append((doc, distance))
    scored.sort(key=lambda item: (item[1], str((item[0].metadata or {}).get("page") or "")))
    return scored[:limit]


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
    """Remove saved PDF bytes from local uploads/ and the durable Blob mirror."""
    settings = get_settings()
    path = settings.uploads_dir / Path(filename).name
    if path.is_file():
        path.unlink()
    try:
        from app.storage.ingest_files import delete_uploaded_pdf_blob

        delete_uploaded_pdf_blob(filename)
    except Exception:
        # Index removal already succeeded; Blob cleanup is best-effort.
        pass
