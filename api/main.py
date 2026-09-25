"""
OpenAI-compatible API — Open WebUI bu endpoint'e bağlanır.

POST /v1/chat/completions  →  RAG pipeline
GET  /v1/models            →  model listesi (Open WebUI için zorunlu)
"""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.pipeline  import RAGPipeline
from rag.situation import DurumAnalizoru
from rag import llm as rag_llm

app = FastAPI(title="BeeChat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton — ilk istekte yüklenir
_rag:    RAGPipeline    | None = None
_analiz: DurumAnalizoru | None = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


def get_analiz() -> DurumAnalizoru:
    global _analiz
    if _analiz is None:
        _analiz = DurumAnalizoru()
    return _analiz


# ── Pydantic modelleri (OpenAI şeması) ───────────────────────────────────── #

class Message(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    model:    str = "beechat"
    messages: list[Message]
    stream:   bool = False


# ── /v1/models — Open WebUI model listesini buradan alır ─────────────────── #

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id":       "beechat",
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "itu",
            },
            {
                "id":       "beechat-analiz",
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "itu",
            },
        ],
    }


# ── /v1/chat/completions ─────────────────────────────────────────────────── #

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages boş olamaz.")

    soru    = req.messages[-1].content.strip()
    analiz_modu = req.model == "beechat-analiz" or soru.lower().startswith("durum:")

    if analiz_modu and soru.lower().startswith("durum:"):
        soru = soru[6:].strip()

    if req.stream:
        return StreamingResponse(
            _stream(soru, analiz_modu),
            media_type="text/event-stream",
        )
    else:
        return await _complete(soru, analiz_modu)


# ── Streaming SSE ─────────────────────────────────────────────────────────── #

async def _stream(soru: str, analiz_modu: bool) -> AsyncIterator[str]:
    import anthropic

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model  = "beechat-analiz" if analiz_modu else "beechat"

    def chunk(delta_content: str, finish: str | None = None) -> str:
        data = {
            "id":      req_id,
            "object":  "chat.completion.chunk",
            "created": int(time.time()),
            "model":   model,
            "choices": [{
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish,
            }],
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        if analiz_modu:
            analizor  = get_analiz()
            faktler   = analizor._fact_extraction(soru)
            queries   = analizor._queries_from_facts(faktler)
            kaynaklar = analizor._multi_retrieve(queries)

            from rag.situation import _ANALYSIS_SYSTEM, _ANALYSIS_USER
            system   = _ANALYSIS_SYSTEM
            user_msg = _ANALYSIS_USER.format(
                ham_giris=soru,
                faktler=analizor._faktler_str(faktler),
                kaynaklar=analizor._build_source_text(kaynaklar),
            )
        else:
            rag      = get_rag()
            chunks   = rag.retriever.retrieve(soru)
            context  = _build_context(chunks)
            system   = rag.cfg.system_prompt
            user_msg = f"Kaynaklar:\n{context}\n\nSoru: {soru}"

        cfg = get_rag().cfg
        for text, kind in rag_llm.stream(system, user_msg, cfg.claude_model, cfg.max_tokens):
            if kind == "text":
                yield chunk(text)

        yield chunk("", finish="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        err = chunk(f"\n\n**Hata:** {e}", finish="stop")
        yield err
        yield "data: [DONE]\n\n"


# ── Non-streaming ─────────────────────────────────────────────────────────── #

async def _complete(soru: str, analiz_modu: bool) -> dict:
    if analiz_modu:
        content = get_analiz().analiz_et(soru).ham_analiz
    else:
        content = get_rag().query(soru).answer

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "beechat",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Yardımcı ─────────────────────────────────────────────────────────────── #

def _build_context(chunks) -> str:
    parts = []
    for rc in chunks:
        m = rc.chunk.metadata
        parts.append(f"[{m['baslik']} — Madde {m['maddeNo']}]\n{rc.chunk.content}")
    return "\n\n---\n\n".join(parts)
