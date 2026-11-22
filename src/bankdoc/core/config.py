"""Application configuration.

All tunable values live in environment variables (or the .env file) and are
loaded into a single typed Settings object. This follows the 12-factor app
principle: configuration is separated from code, so the same code runs in
dev, CI and production with different settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Each field is read from an environment variable
    with the same name (case-insensitive). Missing required values raise
    a clear validation error at startup instead of failing later."""

    # Tell pydantic to also read the local .env file.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    database_url: str

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- LLM provider (used in Step 3) ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # --- Retrieval ---
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    lru_cache makes this a singleton: the .env file is parsed only once,
    and every module that calls get_settings() receives the same object.
    """
    return Settings()
