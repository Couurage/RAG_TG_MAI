from __future__ import annotations

import os
import time
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    GIGACHAT_BASIC,
    GIGACHAT_API_BASE,
    GIGACHAT_OAUTH_URL,
    GIGACHAT_SCOPE,
    GIGACHAT_MODEL,
    GIGACHAT_VERIFY_SSL,
)

load_dotenv()


class GigaChatLLMError(RuntimeError):
    pass


def _parse_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class GigaChatClient:
    API_BASE = GIGACHAT_API_BASE

    OAUTH_URL = GIGACHAT_OAUTH_URL
    SCOPE = GIGACHAT_SCOPE

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        temperature: float = 0.25,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_sec: float = 0.5,
        verify_ssl: bool = True,
    ):
        self.credentials = api_key or GIGACHAT_BASIC
        if not self.credentials:
            raise GigaChatLLMError("Не найден GIGACHAT_BASIC (Authorization key base64).")

        self.model: str = (model or GIGACHAT_MODEL or "GigaChat-2").strip()
        self.temperature: float = float(temperature)
        self.timeout: float = float(timeout)
        self.retries: int = max(0, int(max_retries))
        self.backoff_sec: float = max(0.0, float(backoff_sec))

        # Если переменная окружения задана — парсим её; иначе уважаем аргумент конструктора.
        env_ssl = GIGACHAT_VERIFY_SSL
        self.verify_ssl: bool = _parse_bool(env_ssl, verify_ssl)

        # Создаём клиента один раз и переиспользуем
        self._client = GigaChat(
            credentials=self.credentials,
            model=self.model,
            scope=self.SCOPE,
            verify_ssl_certs=self.verify_ssl,
            timeout=self.timeout,
            temperature=self.temperature,
            profanity_check=False,
            streaming=False,
        )

    def get_answer(self, system_prompt: str, user_prompt: str) -> str:
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                res = self._client.invoke(msgs)
                return (res.content or "").strip()
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    wait = self.backoff_sec * (attempt + 1)  # линейный backoff как у тебя
                    print(f"[WARN] Ошибка при запросе к GigaChat: {e}. Повтор через {wait:.1f} сек...")
                    time.sleep(wait)
                else:
                    break

        raise GigaChatLLMError(f"GigaChat окончательно упал: {last_err}")
