"""
LLM soyutlama katmanı.
LLM_PROVIDER=anthropic  → Anthropic API (varsayılan)
LLM_PROVIDER=openai     → vLLM, Ollama, LM Studio, herhangi OpenAI-compatible endpoint
"""
from __future__ import annotations

import os
from typing import Iterator


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def _openai_client():
    from openai import OpenAI
    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1"),
        api_key=os.environ.get("LLM_API_KEY", "dummy"),
    )


def _model(cfg_model: str) -> str:
    """Ortam değişkeni varsa onu kullan, yoksa config'deki modeli."""
    return os.environ.get("LLM_MODEL", cfg_model)


# ── Streaming ─────────────────────────────────────────────────────────────── #

def stream(system: str, user: str, cfg_model: str, max_tokens: int) -> Iterator[str]:
    """Token token metin üretir."""
    if _provider() == "anthropic":
        import anthropic
        with anthropic.Anthropic().messages.stream(
            model=_model(cfg_model),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as s:
            yield from s.text_stream
    else:
        client = _openai_client()
        with client.chat.completions.create(
            model=_model(cfg_model),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            stream=True,
        ) as s:
            for chunk in s:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ── Tek seferlik ──────────────────────────────────────────────────────────── #

def complete(system: str, user: str, cfg_model: str, max_tokens: int) -> str:
    if _provider() == "anthropic":
        import anthropic
        resp = anthropic.Anthropic().messages.create(
            model=_model(cfg_model),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    else:
        client = _openai_client()
        resp   = client.chat.completions.create(
            model=_model(cfg_model),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content
