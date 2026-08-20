"""
FAISS vector store helpers.

Same building blocks as your LangChain RAG.py: embed texts → FAISS →
as_retriever. Persistence path comes from Settings so uploads survive restarts.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.llm.openrouter import get_embeddings
from app.rag.chunking import TextChunk


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


def build_or_merge_store(chunks: list[TextChunk]) -> FAISS:
    """Create a new index or merge into the existing one on disk."""
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
    return store


def load_store() -> FAISS | None:
    settings = get_settings()
    if not _index_exists(settings.faiss_dir):
        return None
    embeddings = get_embeddings()
    return FAISS.load_local(str(settings.faiss_dir), embeddings, allow_dangerous_deserialization=True)


def _index_exists(path: Path) -> bool:
    return (path / "index.faiss").exists()
