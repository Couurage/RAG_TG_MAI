from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile

from backend.api import deps, schemas
from backend.utils.file_pipeline import RAGPipeline
from backend.utils.rag_service import RAGService

log = logging.getLogger("rag_api")

app = FastAPI(
    title="RAG_TG_MAI API",
    description="HTTP-обёртка над пайплайном RAG для последующей интеграции в Telegram-бота.",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    log.info("Initializing RAG services...")
    deps.init_services()
    log.info("RAG services ready.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    log.info("Shutting down RAG services...")
    deps.shutdown_services()


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Пинг для проверки готовности сервиса."""
    return {"status": "ok"}


@app.post("/query", response_model=schemas.QueryResponse)
async def query_rag(
    payload: schemas.QueryRequest,
    rag_service: RAGService = Depends(deps.get_rag_service),
) -> schemas.QueryResponse:
    """Получить ответ от RAG, используя Milvus + LLM."""
    answer, hits = rag_service.answer(
        payload.question,
        top_k=payload.top_k,
        doc_id=payload.doc_id,
        source_path=payload.source_path,
        owner_id=payload.owner_id,
    )
    formatted_hits = [schemas.Hit(**hit) for hit in hits]
    return schemas.QueryResponse(answer=answer, hits=formatted_hits)


@app.post("/index", response_model=schemas.IndexResponse)
async def index_document(
    file: UploadFile = File(...),
    section: str = Form("default"),
    owner_id: Optional[int] = Form(None),
    pipeline: RAGPipeline = Depends(deps.get_rag_pipeline),
) -> schemas.IndexResponse:
    """Принять файл от клиента, временно сохранить его и передать в пайплайн."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Файл пустой")

    suffix = Path(file.filename or "uploaded").suffix or ".tmp"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(contents)

    try:
        result = pipeline.index_file(tmp_path, section=section, owner_id=owner_id)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not result.chunk_ids:
        raise HTTPException(
            status_code=400,
            detail="Файл не дал ни одного чанка — убедитесь, что в нём есть текст.",
        )

    log.info("Indexed %s chunks from %s (doc_id=%s)", len(result.chunk_ids), file.filename, result.doc_id)
    return schemas.IndexResponse(
        chunks_indexed=len(result.chunk_ids),
        ids=result.chunk_ids,
        section=section,
        doc_id=result.doc_id,
        owner_id=result.owner_id,
    )


@app.delete("/documents", response_model=schemas.DeleteResponse)
async def delete_document(
    doc_id: Optional[int] = Query(None, description="doc_id документа для удаления"),
    source_path: Optional[str] = Query(None, description="Путь к исходному файлу в хранилище"),
    owner_id: Optional[int] = Query(None, description="Владелец документа"),
    pipeline: RAGPipeline = Depends(deps.get_rag_pipeline),
) -> schemas.DeleteResponse:
    """
    Удалить документ из Milvus по doc_id или source_path.
    """
    if doc_id is None and source_path is None:
        raise HTTPException(status_code=400, detail="Нужно указать doc_id или source_path.")

    try:
        removed = pipeline.remove_document(doc_id=doc_id, source_path=source_path, owner_id=owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if removed == 0:
        raise HTTPException(status_code=404, detail="Документ не найден.")

    return schemas.DeleteResponse(deleted_chunks=removed)
