"""
OpenRouter client helpers.

Keeps API key + base_url in one place so agents and RAG share the same
ChatOpenAI setup. Embeddings can swap to a local HF model via
EMBEDDINGS_BACKEND=local (see app/rag/embeddings_local.py).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings

ModelTier = Literal["standard", "advanced"]


def normalize_model_tier(tier: str | None) -> ModelTier:
    raw = (tier or "standard").strip().lower()
    return "advanced" if raw == "advanced" else "standard"


def resolve_chat_model_id(tier: str | None = "standard") -> str:
    """OpenRouter model id for Ask AI documents mode."""
    s = get_settings()
    if normalize_model_tier(tier) == "advanced":
        return (s.openrouter_advanced_model or s.openrouter_model).strip()
    return (s.openrouter_model or "openai/gpt-4o-mini").strip()


def get_chat_model(tier: str | None = "standard") -> ChatOpenAI:
    """Chat model via OpenRouter (OpenAI-compatible)."""
    s = get_settings()
    return ChatOpenAI(
        model=resolve_chat_model_id(tier),
        api_key=s.openrouter_api_key or "missing-key",
        base_url=s.openrouter_base_url,
        temperature=0.2,
    )


def get_embeddings() -> Embeddings:
    """
    Embeddings for FAISS.

    Default: OpenRouter. Set EMBEDDINGS_BACKEND=local for Hugging Face
    sentence-transformers (requires optional install).
    """
    s = get_settings()
    backend = (s.embeddings_backend or "openrouter").strip().lower()

    if backend == "local":
        from app.rag.embeddings_local import get_local_embeddings

        return get_local_embeddings(s.local_embedding_model)

    return OpenAIEmbeddings(
        model=s.openrouter_embedding_model,
        api_key=s.openrouter_api_key or "missing-key",
        base_url=s.openrouter_base_url,
    )
