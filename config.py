from __future__ import annotations

import os
from typing import Final

MILVUS_HOST: Final[str] = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: Final[str] = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB_NAME: Final[str] = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION: Final[str] = os.getenv("MILVUS_COLLECTION", "rag_chunks")
EMBED_DIM: Final[int] = int(os.getenv("EMBED_DIM", "768"))

GIGACHAT_BASIC: Final[str | None] = os.getenv("GIGACHAT_BASIC")
GIGACHAT_API_BASE: Final[str] = os.getenv(
    "GIGACHAT_API_BASE", "https://gigachat.devices.sberbank.ru/api"
)
GIGACHAT_OAUTH_URL: Final[str] = os.getenv(
    "GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
)
GIGACHAT_SCOPE: Final[str] = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL: Final[str] = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
GIGACHAT_VERIFY_SSL: Final[str | None] = os.getenv("GIGACHAT_VERIFY_SSL")
