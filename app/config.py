"""
Load settings from .env.

Design choice: one Settings object so every module reads the same keys.
OpenRouter speaks the OpenAI API shape, so LangChain's ChatOpenAI works
if we set base_url (see app/llm/openrouter.py).
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    if os.environ.get("VERCEL"):
        return Path("/tmp/case-law-agent-data")
    return Path("./data")


class Settings(BaseSettings):
    # Both files are read, with .env.local winning. `vercel env pull` writes
    # .env.local, so Neon and Blob credentials land there, while .env stays the
    # hand-edited file for model choices. Later entries take precedence in
    # pydantic-settings.
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    # Direct OpenAI Responses API for Ask AI Web mode. The existing
    # OpenRouter key remains the fallback and still powers PDF embeddings.
    openai_api_key: str = ""
    openai_web_model: str = "gpt-5-mini"
    # openrouter (default) | local
    embeddings_backend: str = "openrouter"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = Path("./data")

    # Neon Postgres. The Vercel Neon integration writes several URLs; we want the
    # pooled one (host contains "-pooler") because serverless functions open and
    # drop connections constantly and Neon's PgBouncer absorbs that churn.
    database_url: str = ""
    # Seconds to wait for a Neon connection. Neon cold-starts a compute in a few
    # hundred milliseconds, so a short timeout still needs headroom.
    database_connect_timeout: int = 10

    # Vercel injects this in deployed functions. Declaring it here as well is
    # what makes Blob work locally, where the value only exists in .env.local.
    blob_read_write_token: str = ""

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url.strip())

    @property
    def web_search_configured(self) -> bool:
        return bool(self.openai_api_key.strip() or self.openrouter_api_key.strip())

    def model_post_init(self, __context) -> None:
        if os.environ.get("VERCEL"):
            object.__setattr__(self, "data_dir", _default_data_dir())

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss_index"

    @property
    def bundled_faiss_dir(self) -> Path:
        """FAISS index baked into the deployment bundle at build time."""
        return _REPO_ROOT / "data" / "faiss_index"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for directory in (settings.uploads_dir, settings.faiss_dir):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return settings
