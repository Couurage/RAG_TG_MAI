from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class RAGClient:
    """HTTP-клиент для общения с FastAPI."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        doc_id: Optional[int] = None,
        owner_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question": question,
            "top_k": top_k,
        }
        if doc_id is not None:
            payload["doc_id"] = doc_id
        if owner_id is not None:
            payload["owner_id"] = owner_id
        resp = await self._client.post("/query", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def index_document(
        self,
        path: Path,
        *,
        section: str,
        owner_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        with path.open("rb") as fh:
            files = {"file": (path.name, fh, "application/octet-stream")}
            data: Dict[str, Any] = {"section": section}
            if owner_id is not None:
                data["owner_id"] = owner_id
            resp = await self._client.post("/index", data=data, files=files)
        resp.raise_for_status()
        return resp.json()

    async def delete_document(
        self,
        *,
        doc_id: Optional[int] = None,
        owner_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if doc_id is None:
            raise ValueError("doc_id должен быть указан для удаления через бот")
        params: Dict[str, Any] = {"doc_id": doc_id}
        if owner_id is not None:
            params["owner_id"] = owner_id
        resp = await self._client.delete("/documents", params=params)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()
