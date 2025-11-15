from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

from backend.database.milvus import Milvus
from backend.gemma_services.tokenizer import Tokenizer
from backend.gemma_services.embedder import Embedder
from backend.utils.markdown import convert_to_md


class RAGPipeline:
    def __init__(
        self,
        milvus: Optional[Milvus] = None,
        tokenizer: Optional[Tokenizer] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.milvus = milvus or Milvus()
        self.tokenizer = tokenizer or Tokenizer()
        self.embedder = embedder or Embedder()

    def _file_to_markdown(self, path: Path) -> tuple[int, str]:
        md_path, doc_id = convert_to_md(path)
        md_text = md_path.read_text(encoding="utf-8")
        return doc_id, md_text

    def index_file(
        self,
        input_path: str | Path,
        section: str = "default",
    ) -> List[int]:
        input_path = Path(input_path)
        doc_id, md_text = self._file_to_markdown(input_path)
        chunks: List[str] = self.tokenizer.chunk_markdown(md_text)

        if not chunks:
            return []

        embeddings: List[List[float]] = [
            self.embedder.embed(chunk) for chunk in chunks
        ]

        sections: List[Optional[str]] = [section] * len(chunks)
        source_path: str = str(input_path)

        ids: List[int] = self.milvus.add_chunks(
            doc_id=doc_id,
            source_path=source_path,
            sections=sections,
            contents=chunks,
            embeddings=embeddings,
        )

        return ids

    def index_folder(
        self,
        folder: str | Path,
        section: str = "default",
        recursive: bool = False,
    ) -> Dict[str, List[int]]:
        folder = Path(folder)
        results: Dict[str, List[int]] = {}

        if not folder.exists():
            raise FileNotFoundError(folder)

        iterable = folder.rglob("*") if recursive else folder.iterdir()

        for path in iterable:
            if not path.is_file():
                continue
            try:
                ids = self.index_file(path, section=section)
                results[path.name] = ids
            except Exception:
                pass

        return results

    def search(self, query: str, top_k: int = 5):
        query_emb: List[float] = self.embedder.embed(query)
        return self.milvus.search(query_emb=query_emb, top_k=top_k)
