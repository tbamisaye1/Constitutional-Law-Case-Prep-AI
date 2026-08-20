"""
Load settings from .env.

Design choice: one Settings object so every module reads the same keys.
OpenRouter speaks the OpenAI API shape, so LangChain's ChatOpenAI works
if we set base_url (see app/llm/openrouter.py).
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    # openrouter (default) | local
    embeddings_backend: str = "openrouter"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = Path("./data")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss_index"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    return settings
