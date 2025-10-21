from __future__ import annotations

import os
from dotenv import load_dotenv
import time

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class GigaChatLLMError(RuntimeError):
    pass


class GigaChatClient:
    API_BASE = os.getenv("GIGACHAT_API_BASE", "https://gigachat.devices.sberbank.ru/api")

    OAUTH_URL = os.getenv("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
    SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

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
        self.credentials = api_key or os.getenv("GIGACHAT_BASIC")
        if not self.credentials:
            raise GigaChatLLMError("Не найден GIGACHAT_BASIC (Authorization key base64).")

        self.model: str = (model or os.getenv("GIGACHAT_MODEL") or "GigaChat-2").strip()
        self.temperature: float = float(temperature)
        self.timeout: int = int(timeout)
        self.retries: int = int(max_retries)
        self.backoff_sec: float = float(backoff_sec)
        self.verify_ssl: bool = os.getenv("GIGACHAT_VERIFY_SSL", False)
