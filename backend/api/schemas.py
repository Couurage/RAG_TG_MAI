from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Hit(BaseModel):
    score: float
    doc_id: Optional[int] = None
    chunk_id: Optional[int] = None
    section: Optional[str] = None
    source_path: Optional[str] = None
    content: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Пользовательский вопрос")
    top_k: int = Field(5, ge=1, le=20, description="Сколько чанков вернуть из Milvus")
    doc_id: Optional[int] = Field(
        None,
        description="Ограничить поиск конкретным doc_id (хэш файла)",
    )
    source_path: Optional[str] = Field(
        None,
        description="Фильтр по исходному пути (как хранится в Milvus)",
    )
    owner_id: Optional[int] = Field(
        None,
        description="Идентификатор владельца документа (для изоляции данных)",
    )


class QueryResponse(BaseModel):
    answer: str
    hits: List[Hit]


class IndexResponse(BaseModel):
    chunks_indexed: int
    ids: List[int]
    section: str
    doc_id: int
    owner_id: Optional[int] = None


class DeleteResponse(BaseModel):
    deleted_chunks: int
