"""
OpenRouter / OpenAI client helpers.

Keeps API key + base_url in one place so agents and RAG share the same
ChatOpenAI setup. Embeddings can swap to a local HF model via
EMBEDDINGS_BACKEND=local (see app/rag/embeddings_local.py).

Ask AI model tiers:
  standard  — OpenRouter OPENROUTER_MODEL (cheap default, gpt-4o-mini)
  advanced  — prefer direct OpenAI OPENAI_ADVANCED_WEB_MODEL (gpt-5-mini)
              when OPENAI_API_KEY is set; else OpenRouter advanced model
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


def _is_gpt5_family(model_id: str) -> bool:
    name = (model_id or "").lower()
    return "gpt-5" in name or "/o1" in name or name.startswith("o1") or name.startswith("o3")


def resolve_chat_model_id(tier: str | None = "standard") -> str:
    """Model id Ask AI will call for this tier (display + invoke)."""
    s = get_settings()
    if normalize_model_tier(tier) == "advanced":
        # Prefer the same OpenAI account Web mode already uses. OpenRouter
        # often 402s on gpt-5-mini when credits are thin.
        if s.openai_api_key.strip():
            return (s.openai_advanced_web_model or s.openai_web_model or "gpt-5-mini").strip()
        return (s.openrouter_advanced_model or s.openrouter_model).strip()
    return (s.openrouter_model or "openai/gpt-4o-mini").strip()


def get_chat_model(tier: str | None = "standard") -> ChatOpenAI:
    """
    Chat model for documents-mode Ask AI.

    Advanced uses direct OpenAI when possible so the toggle does not depend
    on OpenRouter having credit for gpt-5-mini.
    """
    s = get_settings()
    tier_n = normalize_model_tier(tier)
    model_id = resolve_chat_model_id(tier_n)

    kwargs: dict = {"model": model_id}
    if not _is_gpt5_family(model_id):
        kwargs["temperature"] = 0.2

    if tier_n == "advanced" and s.openai_api_key.strip():
        kwargs["api_key"] = s.openai_api_key
        return ChatOpenAI(**kwargs)

    kwargs["api_key"] = s.openrouter_api_key or "missing-key"
    kwargs["base_url"] = s.openrouter_base_url
    return ChatOpenAI(**kwargs)


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
