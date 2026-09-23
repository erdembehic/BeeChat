"""End-to-end RAG pipeline."""
import anthropic

from .config import Config, cfg as default_cfg
from .retriever import Retriever
from .models import RAGResult, RetrievedChunk


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for rc in chunks:
        m = rc.chunk.metadata
        source_tag = "ref_expansion" if rc.source == "ref_expansion" else ""
        header = f"[{m['baslik']} — Madde {m['maddeNo']}]"
        if source_tag:
            header += " (çapraz referans)"
        parts.append(f"{header}\n{rc.chunk.content}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    def __init__(self, cfg: Config = default_cfg):
        self.cfg = cfg
        self.retriever = Retriever(cfg)
        self.client = anthropic.Anthropic()

    def query(self, soru: str, stream: bool = False) -> RAGResult:
        # 1. Retrieval
        chunks = self.retriever.retrieve(soru)

        if not chunks:
            return RAGResult(
                answer="İlgili mevzuat bulunamadı.",
                chunks=[],
                query=soru,
            )

        # 2. Context
        context = _build_context(chunks)
        user_msg = f"Kaynaklar:\n{context}\n\nSoru: {soru}"

        # 3. Generation
        if stream:
            answer_parts = []
            with self.client.messages.stream(
                model=self.cfg.claude_model,
                max_tokens=self.cfg.max_tokens,
                system=self.cfg.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            ) as s:
                for text in s.text_stream:
                    print(text, end="", flush=True)
                    answer_parts.append(text)
            print()
            answer = "".join(answer_parts)
        else:
            response = self.client.messages.create(
                model=self.cfg.claude_model,
                max_tokens=self.cfg.max_tokens,
                system=self.cfg.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            answer = response.content[0].text

        return RAGResult(answer=answer, chunks=chunks, query=soru)

    def query_with_filter(
        self,
        soru: str,
        sadece_yonetmelik: bool = False,
        mevzuat_no: str | None = None,
    ) -> RAGResult:
        """Belirli bir yönetmelik veya türe göre filtreli sorgu."""
        # Retriever'a filtre uygula (Qdrant payload filter)
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if sadece_yonetmelik:
            conditions.append(FieldCondition(key="tur", match=MatchValue(value="yonetmelik")))
        if mevzuat_no:
            conditions.append(FieldCondition(key="mevzuatNo", match=MatchValue(value=mevzuat_no)))

        _orig_search = self.retriever.hybrid_search

        if conditions:
            qfilter = Filter(must=conditions)

            def filtered_search(query: str):
                from qdrant_client.models import (
                    PrefetchQuery, NamedVector, NamedSparseVector,
                    FusionQuery, Fusion,
                )
                dense_vec  = self.retriever._dense_vec(query)
                sparse_vec = self.retriever._sparse_vec(query)
                results = self.retriever.client.query_points(
                    collection_name=self.cfg.collection_name,
                    prefetch=[
                        PrefetchQuery(
                            query=NamedVector(name="dense", vector=dense_vec),
                            limit=self.cfg.retrieve_top_k,
                            filter=qfilter,
                        ),
                        PrefetchQuery(
                            query=NamedSparseVector(
                                name="sparse", vector=sparse_vec,
                            ),
                            limit=self.cfg.retrieve_top_k,
                            filter=qfilter,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=self.cfg.retrieve_top_k,
                    with_payload=True,
                )
                return [
                    self.retriever._payload_to_chunk(r.payload, r.score, "hybrid")
                    for r in results.points
                ]

            self.retriever.hybrid_search = filtered_search

        result = self.query(soru)
        self.retriever.hybrid_search = _orig_search
        return result
