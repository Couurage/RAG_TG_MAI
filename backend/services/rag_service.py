from __future__ import annotations

from typing import List, Tuple, Optional, Dict

from backend.utils.file_pipeline import RAGPipeline
from backend.models.giga import GigaChatClient


DEFAULT_SYSTEM_PROMPT = (
    "Ты — ассистент RAG-системы. "
    "Отвечай только на основе переданного контекста. "
    "Если информации недостаточно, честно скажи об этом. "
    "Не придумывай факты и не добавляй сведения, которых нет в контексте. "
    "Если фрагменты противоречат друг другу, укажи на это. "
    "Отвечай по-русски, кратко, структурированно и логично."
)


class RAGService:
    def __init__(
        self,
        pipeline: Optional[RAGPipeline] = None,
        llm: Optional[GigaChatClient] = None,
        *,
        context_limit: int = 5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.pipeline = pipeline or RAGPipeline()
        self.llm = llm or GigaChatClient()
        self.context_limit = max(1, context_limit)
        self.system_prompt = system_prompt

    def _format_context(self, hits: List[Dict]) -> str:
        chunks: List[str] = []
        for idx, hit in enumerate(hits[: self.context_limit], 1):
            content = (hit.get("content") or "").strip()
            if not content:
                continue
            source = hit.get("source_path") or "unknown"
            chunk_id = hit.get("chunk_id")
            prefix = f"[{idx}] source={source}"
            if chunk_id is not None:
                prefix += f" chunk={chunk_id}"
            chunks.append(f"{prefix}\n{content}")
        return "\n\n".join(chunks).strip()

    def _build_user_prompt(self, question: str, context: str) -> str:
        return (
            f"Контекст:\n{context}\n\n"
            f"Вопрос пользователя:\n{question}\n\n"
            "Сформулируй понятный ответ, обязательно опираясь на контекст выше."
        )

    def answer(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        hits = self.pipeline.search(question, top_k=top_k)
        if not hits:
            return "", []

        context = self._format_context(hits)
        if not context:
            return "", hits

        user_prompt = self._build_user_prompt(question, context)
        answer = self.llm.get_answer(self.system_prompt, user_prompt)
        return answer, hits

