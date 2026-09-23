"""Voyage AI dense embedding — lokal model yok, saf API."""
from __future__ import annotations

import os
import time
from typing import Sequence

import voyageai

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise EnvironmentError("VOYAGE_API_KEY ortam değişkeni eksik.")
        _client = voyageai.Client(api_key=api_key)
    return _client


MODEL = "voyage-multilingual-2"
DIM   = 1024
BATCH = 128  # Voyage max batch


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Passage embedding — index'leme için."""
    client = _get_client()
    result: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp  = client.embed(batch, model=MODEL, input_type="document")
        result.extend(resp.embeddings)
        if i + BATCH < len(texts):
            time.sleep(0.05)  # rate limit marjı
    return result


def embed_query(text: str) -> list[float]:
    """Query embedding — sorgulama için."""
    client = _get_client()
    resp   = _get_client().embed([text], model=MODEL, input_type="query")
    return resp.embeddings[0]
