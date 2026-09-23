"""Cohere Rerank API — lokal cross-encoder yok."""
from __future__ import annotations

import os
import cohere

_client: cohere.Client | None = None


def _get_client() -> cohere.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError("COHERE_API_KEY ortam değişkeni eksik.")
        _client = cohere.Client(api_key=api_key)
    return _client


MODEL = "rerank-multilingual-v3"


def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """
    (orijinal_index, relevance_score) listesi döndürür, skora göre sıralı.
    documents listesi boşsa boş liste döner.
    """
    if not documents:
        return []

    results = _get_client().rerank(
        query=query,
        documents=documents,
        model=MODEL,
        top_n=min(top_n, len(documents)),
    )
    return [(r.index, r.relevance_score) for r in results.results]
