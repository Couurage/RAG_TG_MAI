from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from backend.database.milvus import Milvus
from backend.gemma_services.tokenizer import Tokenizer
from backend.gemma_services.embedder import Embedder
from backend.utils.markdown import convert_to_md


log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    doc_id: int
    chunk_ids: List[int]
    source_path: str
    owner_id: Optional[int] = None


class RAGPipeline:
    def __init__(
        self,
        milvus: Optional[Milvus] = None,
        tokenizer: Optional[Tokenizer] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.embedder = embedder or Embedder()
        embed_dim = self.embedder.dim()
        self.milvus = milvus or Milvus(embed_dim=embed_dim)
        self.tokenizer = tokenizer or Tokenizer()

    def _file_to_markdown(self, path: Path) -> tuple[int, str]:
        md_path, doc_id = convert_to_md(path)
        md_text = md_path.read_text(encoding="utf-8")
        return doc_id, md_text

    def index_file(
        self,
        input_path: str | Path,
        section: str = "default",
        *,
        owner_id: Optional[int] = None,
    ) -> IndexResult:
        input_path = Path(input_path)
        doc_id, md_text = self._file_to_markdown(input_path)
        chunks: List[str] = self.tokenizer.chunk_markdown(md_text)

        source_path: str = str(input_path)
        if not chunks:
            return IndexResult(doc_id=doc_id, chunk_ids=[], source_path=source_path, owner_id=owner_id)

        removed = self.milvus.delete_doc(doc_id=doc_id, owner_id=owner_id)
        if removed:
            log.info("Removed %s old chunks for doc_id=%s (%s)", removed, doc_id, input_path)

        embeddings: List[List[float]] = self.embedder.embed(chunks)

        sections: List[Optional[str]] = [section] * len(chunks)
        owner_value = int(owner_id) if owner_id is not None else -1

        ids: List[int] = self.milvus.add_chunks(
            owner_id=owner_value,
            doc_id=doc_id,
            source_path=source_path,
            sections=sections,
            contents=chunks,
            embeddings=embeddings,
        )

        return IndexResult(doc_id=doc_id, chunk_ids=ids, source_path=source_path, owner_id=owner_value)

    def index_folder(
        self,
        folder: str | Path,
        section: str = "default",
        recursive: bool = False,
    ) -> Dict[str, IndexResult]:
        folder = Path(folder)
        results: Dict[str, IndexResult] = {}

        if not folder.exists():
            raise FileNotFoundError(folder)

        iterable = folder.rglob("*") if recursive else folder.iterdir()

        for path in iterable:
            if not path.is_file():
                continue
            try:
                result = self.index_file(path, section=section)
                results[str(path)] = result
            except Exception as exc:
                log.exception("Failed to index %s: %s", path, exc)

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        doc_id: Optional[int] = None,
        source_path: Optional[str] = None,
        owner_id: Optional[int] = None,
    ):
        query_embs = self.embedder.embed([query])
        if not query_embs:
            raise ValueError("Query produced no embedding")
        query_emb: List[float] = query_embs[0]
        hits = self.milvus.search(
            query_emb=query_emb,
            top_k=top_k,
            doc_id=doc_id,
            source_path=source_path,
            owner_id=owner_id,
        )

        formatted = []
        for hit in hits:
            entity = hit.entity
            formatted.append(
                {
                    "score": float(hit.distance),
                    "doc_id": entity.get("doc_id"),
                    "chunk_id": entity.get("chunk_id"),
                    "section": entity.get("section"),
                    "source_path": entity.get("source_path"),
                    "content": entity.get("content"),
                }
            )
        return formatted

    def remove_document(
        self,
        *,
        doc_id: Optional[int] = None,
        source_path: Optional[str] = None,
        owner_id: Optional[int] = None,
    ) -> int:
        """
        Удаляет документ из Milvus по doc_id или source_path.
        Возвращает количество удалённых чанков.
        """
        if doc_id is None and source_path is None:
            raise ValueError("doc_id or source_path must be provided")

        removed = self.milvus.delete_doc(doc_id=doc_id, source_path=source_path, owner_id=owner_id)
        if removed:
            log.info(
                "Removed %s chunks for doc_id=%s source_path=%s owner_id=%s",
                removed,
                doc_id,
                source_path,
                owner_id,
            )
        return removed
