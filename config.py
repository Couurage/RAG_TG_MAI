

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "var"
DEFAULT_MILVUS_PATH = DEFAULT_DATA_DIR / "milvus" / "milvus.db"
DEFAULT_UPLOADS_DIR = DEFAULT_DATA_DIR / "uploads"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Settings(BaseSettings):
    """Глобальные настройки Backend RAG (читаются из .env и окружения)."""

    # -------------------- Embedding service (FastAPI /embed) --------------------
    embedding_model_name: str = Field(
        default="jinaai/jina-embeddings-v3",
        env="EMBEDDING_MODEL_NAME",
    )
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    huggingface_hub_token: Optional[str] = Field(default=None, env="HUGGINGFACE_HUB_TOKEN")

    @computed_field  # type: ignore[misc]
    @property
    def huggingface_token(self) -> Optional[str]:
        return self.hf_token or self.huggingface_hub_token

    @field_validator("milvus_uri", mode="before")
    @classmethod
    def _resolve_milvus_uri(cls, value: object) -> str:
        """Путь к локальной БД приводим к абсолютному и создаём директорию."""
        if not value:
            value = str(DEFAULT_MILVUS_PATH)
        if isinstance(value, Path):
            value = str(value)

        # Если это URI (например, http:// или sqlite://), не трогаем
        if "://" in value:
            return value

        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        _ensure_dir(path.parent)
        return str(path)

    @field_validator("uploads_dir", mode="before")
    @classmethod
    def _resolve_uploads_dir(cls, value: object) -> str:
        if not value:
            value = str(DEFAULT_UPLOADS_DIR)
        if isinstance(value, Path):
            value = str(value)
        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        _ensure_dir(path)
        return str(path)


    # -------------------- GigaChat (для генерации ответа) --------------------
    gigachat_api_base: str = Field(default="https://gigachat.devices.sberbank.ru/api", env="GIGACHAT_API_BASE")
    gigachat_oauth_url: str = Field(default="https://ngw.devices.sberbank.ru:9443/api/v2/oauth", env="GIGACHAT_OAUTH_URL")
    gigachat_scope: str = Field(default="GIGACHAT_API_PERS", env="GIGACHAT_SCOPE")
    gigachat_model: str = Field(default="GigaChat-2", env="GIGACHAT_MODEL")
    # base64 в Authorization для Basic (см. твой GigaChatClient)
    gigachat_basic: Optional[str] = Field(default=None, env="GIGACHAT_BASIC")
    # отключать ли проверку SSL (в dev-средах)
    gigachat_verify_ssl: bool = Field(default=False, env="GIGACHAT_VERIFY_SSL")

    # -------------------- Pydantic Settings --------------------
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Инициализируем единожды
settings = Settings()



# Embedding service (совместимость)
SERVICE_URL: str = "local"


# GigaChat (если нужно обращаться напрямую из других модулей)
GIGACHAT_API_BASE: str = settings.gigachat_api_base
GIGACHAT_OAUTH_URL: str = settings.gigachat_oauth_url
GIGACHAT_SCOPE: str = settings.gigachat_scope
GIGACHAT_MODEL: str = settings.gigachat_model
GIGACHAT_BASIC: Optional[str] = settings.gigachat_basic
GIGACHAT_VERIFY_SSL: bool = settings.gigachat_verify_ssl

SERVICE_URL: str = "local"

GIGACHAT_API_BASE: str = settings.gigachat_api_base
GIGACHAT_OAUTH_URL: str = settings.gigachat_oauth_url
GIGACHAT_SCOPE: str = settings.gigachat_scope
GIGACHAT_MODEL: str = settings.gigachat_model
GIGACHAT_BASIC: Optional[str] = settings.gigachat_basic
GIGACHAT_VERIFY_SSL: bool = settings.gigachat_verify_ssl
