"""
Application configuration.

All settings are read from environment variables (or .env file).
Pydantic Settings validates them at startup — if a required value is missing,
the app crashes immediately with a clear error instead of failing later.

Usage:
    from app.config import settings
    print(settings.database_url)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the entire application.

    Every field here maps to an environment variable with the same name (uppercase).
    For example, `database_url` reads from `DATABASE_URL`.

    Fields with default values are optional in .env.
    Fields without defaults are REQUIRED — the app won't start without them.
    """

    # --- Application ---
    app_name: str = "Support AI"
    environment: str = "local"  # local | staging | production

    # --- Auth ---
    api_key: str  # REQUIRED — the X-API-Key value clients must send

    # --- Database ---
    database_url: str  # REQUIRED — e.g. postgresql://user:pass@localhost:5432/dbname

    # --- Gemini LLM ---
    gemini_api_key: str  # REQUIRED — from Google AI Studio
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # --- File Storage ---
    knowledge_base_path: str = "data/knowledge_base"  # local dev file storage

    # --- Pydantic Settings config ---
    model_config = SettingsConfigDict(
        env_file=".env",           # load from .env file
        env_file_encoding="utf-8",
        extra="ignore",            # ignore env vars we don't define here
    )


# Singleton instance — import this everywhere
settings = Settings()
