"""
LLM soyutlama katmanı — tüm major provider'lar desteklenir.

LLM_PROVIDER seçenekleri:
  anthropic   → Anthropic API (varsayılan)
  nvidia      → NVIDIA NIM (integrate.api.nvidia.com)
  groq        → Groq API (çok hızlı, ücretsiz tier)
  together    → Together AI
  deepseek    → DeepSeek (R1 thinking desteği ile)
  mistral     → Mistral AI
  openai      → OpenAI veya herhangi OpenAI-compatible (vLLM, Ollama, LM Studio)

Tüm non-anthropic provider'lar OpenAI-compatible API kullanır.
Thinking token'ları (DeepSeek R1, QwQ) otomatik ayrıştırılır.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterator


# ── Provider presets ──────────────────────────────────────────────────────── #

PRESETS: dict[str, dict] = {
    "anthropic": {},
    "nvidia":    {
        "base_url":      "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama-3.1-70b-instruct",
    },
    "groq": {
        "base_url":      "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-70b-versatile",
    },
    "together": {
        "base_url":      "https://api.together.xyz/v1",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    },
    "deepseek": {
        "base_url":      "https://api.deepseek.com/v1",
        "default_model": "deepseek-reasoner",   # R1 — thinking desteği var
    },
    "mistral": {
        "base_url":      "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
    },
    "openai": {
        "base_url":      "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
}


# ── Yardımcılar ───────────────────────────────────────────────────────────── #

def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def _env(key: str, fallback: str = "") -> str:
    return os.environ.get(key, fallback)


def _openai_client(provider: str):
    from openai import OpenAI
    preset   = PRESETS.get(provider, {})
    base_url = _env("LLM_BASE_URL", preset.get("base_url", ""))
    api_key  = _env("LLM_API_KEY") or _env(f"{provider.upper()}_API_KEY", "dummy")
    return OpenAI(base_url=base_url, api_key=api_key)


def _resolve_model(cfg_model: str, provider: str) -> str:
    """LLM_MODEL env > config model > provider default."""
    if m := _env("LLM_MODEL"):
        return m
    if provider != "anthropic" and cfg_model.startswith("claude-"):
        return PRESETS.get(provider, {}).get("default_model", cfg_model)
    return cfg_model


# ── Thinking token ayrıştırma ─────────────────────────────────────────────── #

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LLMResponse:
    text:    str             # nihai cevap
    thinking: str | None     # <think> bloğu veya Claude extended thinking


def _split_thinking(raw: str) -> LLMResponse:
    """DeepSeek R1 / QwQ stil <think> tag'lerini ayır."""
    thoughts = _THINK_RE.findall(raw)
    answer   = _THINK_RE.sub("", raw).strip()
    return LLMResponse(
        text=answer,
        thinking="\n\n".join(thoughts).strip() or None,
    )


# ── Streaming ─────────────────────────────────────────────────────────────── #

def stream(
    system: str,
    user: str,
    cfg_model: str,
    max_tokens: int,
) -> Iterator[tuple[str, str]]:
    """
    (token, kind) çiftleri üretir.
    kind: "text" | "thinking"
    Tüketici sadece "text" istiyorsa kind'ı görmezden gelebilir.
    """
    provider = _provider()
    model    = _resolve_model(cfg_model, provider)

    if provider == "anthropic":
        yield from _stream_anthropic(system, user, model, max_tokens)
    else:
        yield from _stream_openai(provider, system, user, model, max_tokens)


def _stream_anthropic(system, user, model, max_tokens) -> Iterator[tuple[str, str]]:
    import anthropic as _ant
    client = _ant.Anthropic()

    # Extended thinking — sadece claude-3-7 ve üstü destekler
    supports_thinking = "claude-3-7" in model or "claude-opus-4" in model
    kwargs: dict = {}
    if supports_thinking and _env("LLM_THINKING", "0") == "1":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
        kwargs["betas"]    = ["interleaved-thinking-2025-05-14"]

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        **kwargs,
    ) as s:
        for event in s:
            import anthropic as _ant2
            if isinstance(event, _ant2.RawContentBlockDeltaEvent):
                delta = event.delta
                if hasattr(delta, "thinking"):
                    yield delta.thinking, "thinking"
                elif hasattr(delta, "text"):
                    yield delta.text, "text"


def _stream_openai(provider, system, user, model, max_tokens) -> Iterator[tuple[str, str]]:
    client = _openai_client(provider)
    buf    = []   # thinking buffer için
    in_think = False

    with client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        stream=True,
    ) as s:
        for chunk in s:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            # <think>...</think> akışlı ayrıştırma
            buf.append(delta)
            combined = "".join(buf)

            if not in_think and "<think>" in combined:
                # <think> öncesi text varsa ver
                before, _, rest = combined.partition("<think>")
                if before.strip():
                    yield before, "text"
                buf = [rest]
                in_think = True
                continue

            if in_think and "</think>" in combined:
                think_part, _, after = combined.partition("</think>")
                yield think_part, "thinking"
                buf = [after]
                in_think = False
                continue

            # Normal akış
            if not in_think:
                yield delta, "text"
                buf = []


# ── Tek seferlik (non-streaming) ──────────────────────────────────────────── #

def complete(system: str, user: str, cfg_model: str, max_tokens: int) -> LLMResponse:
    """LLMResponse döndürür — .text nihai cevap, .thinking düşünme adımları."""
    provider = _provider()
    model    = _resolve_model(cfg_model, provider)

    if provider == "anthropic":
        import anthropic as _ant
        resp = _ant.Anthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(text=resp.content[-1].text, thinking=None)
    else:
        client = _openai_client(provider)
        resp   = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return _split_thinking(resp.choices[0].message.content or "")


# ── Geriye uyumluluk: sadece text isteyen çağrıcılar için ─────────────────── #

def stream_text(system: str, user: str, cfg_model: str, max_tokens: int) -> Iterator[str]:
    for token, kind in stream(system, user, cfg_model, max_tokens):
        if kind == "text":
            yield token


def complete_text(system: str, user: str, cfg_model: str, max_tokens: int) -> str:
    return complete(system, user, cfg_model, max_tokens).text
