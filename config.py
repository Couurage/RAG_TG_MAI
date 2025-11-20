from __future__ import annotations

import os
from typing import Final

MILVUS_HOST: Final[str] = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: Final[str] = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB_NAME: Final[str] = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION: Final[str] = os.getenv("MILVUS_COLLECTION", "rag_chunks")
EMBED_DIM: Final[int] = int(os.getenv("EMBED_DIM", "768"))
