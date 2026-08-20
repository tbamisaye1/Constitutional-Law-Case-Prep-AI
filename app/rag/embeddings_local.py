"""
Local Hugging Face embeddings.

Same LangChain Embeddings interface as OpenRouter, so FAISS code in store.py
does not care which backend you pick.

Install when using EMBEDDINGS_BACKEND=local:
  pip install sentence-transformers

Default model (CPU-friendly): BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def get_local_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> Embeddings:
    """
    Lazy-load a sentence-transformers model via LangChain's HuggingFaceEmbeddings.

    Raises a clear error if the optional dependency is missing, so OpenRouter
    users are not forced to install torch on day one.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Local embeddings need langchain-community + sentence-transformers. "
            "pip install sentence-transformers  "
            "Or set EMBEDDINGS_BACKEND=openrouter"
        ) from exc

    # normalize_embeddings helps cosine / FAISS L2 behavior stay sensible
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
