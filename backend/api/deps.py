from __future__ import annotations

from typing import Optional

from fastapi import Depends

from backend.utils.file_pipeline import RAGPipeline
from backend.utils.rag_service import RAGService

_pipeline: Optional[RAGPipeline] = None
_rag_service: Optional[RAGService] = None


def init_services() -> None:
    global _pipeline, _rag_service
    if _pipeline is None:
        _pipeline = RAGPipeline()
    if _rag_service is None:
        _rag_service = RAGService(pipeline=_pipeline)


def shutdown_services() -> None:
    global _pipeline, _rag_service
    _rag_service = None
    _pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("RAG pipeline is not initialized yet")
    return _pipeline


def get_rag_service(pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> RAGService:
    if _rag_service is None:
        raise RuntimeError("RAG service is not initialized yet")
    return _rag_service

