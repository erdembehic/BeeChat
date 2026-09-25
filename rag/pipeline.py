"""End-to-end RAG pipeline."""
from .config import Config, cfg as default_cfg
from .retriever import Retriever
from .models import RAGResult, RetrievedChunk
from . import llm


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for rc in chunks:
        m      = rc.chunk.metadata
        header = f"[{m['baslik']} — Madde {m['maddeNo']}]"
        if rc.source == "ref_expansion":
            header += " (çapraz referans)"
        parts.append(f"{header}\n{rc.chunk.content}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    def __init__(self, cfg: Config = default_cfg):
        self.cfg = cfg
        self.retriever = Retriever(cfg)

    def query(self, soru: str, stream: bool = False) -> RAGResult:
        chunks = self.retriever.retrieve(soru)
        if not chunks:
            return RAGResult(answer="İlgili mevzuat bulunamadı.", chunks=[], query=soru)

        context  = _build_context(chunks)
        user_msg = f"Kaynaklar:\n{context}\n\nSoru: {soru}"

        if stream:
            parts = []
            for text, kind in llm.stream(self.cfg.system_prompt, user_msg,
                                         self.cfg.claude_model, self.cfg.max_tokens):
                if kind == "text":
                    print(text, end="", flush=True)
                    parts.append(text)
            print()
            answer = "".join(parts)
        else:
            answer = llm.complete(self.cfg.system_prompt, user_msg,
                                  self.cfg.claude_model, self.cfg.max_tokens).text

        return RAGResult(answer=answer, chunks=chunks, query=soru)

    def query_with_filter(
        self,
        soru: str,
        sadece_yonetmelik: bool = False,
        mevzuat_no: str | None = None,
    ) -> RAGResult:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if sadece_yonetmelik:
            conditions.append(FieldCondition(key="tur", match=MatchValue(value="yonetmelik")))
        if mevzuat_no:
            conditions.append(FieldCondition(key="mevzuatNo", match=MatchValue(value=mevzuat_no)))

        if conditions:
            qfilter = Filter(must=conditions)
            _orig   = self.retriever.dense_search

            def filtered(query, limit=None, **_):
                return _orig(query, limit=limit, qfilter=qfilter)

            self.retriever.dense_search = filtered
            result = self.query(soru)
            self.retriever.dense_search = _orig
        else:
            result = self.query(soru)

        return result
