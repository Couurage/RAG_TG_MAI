from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    rag_api_base: str = Field("http://localhost:8000", alias="RAG_API_BASE")
    default_top_k: int = Field(5, alias="RAG_BOT_TOP_K")
    default_section: str = Field("telegram", alias="RAG_BOT_SECTION")
    request_timeout: float = Field(1800.0, alias="RAG_BOT_TIMEOUT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")


settings = Settings()

