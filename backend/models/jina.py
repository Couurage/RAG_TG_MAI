#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Совместимый клиент для локального сервиса эмбеддингов."""

from __future__ import annotations

from typing import Dict, List

from backend.services.embedding import embedding_service


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Возвращает эмбеддинги текстов, не делая HTTP-запросов."""

    return embedding_service.embed(texts)


def health_check() -> Dict[str, str | int]:
    """Информация о состоянии локальной модели."""

    return embedding_service.health()
