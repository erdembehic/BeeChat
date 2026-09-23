"""Hybrid retrieval + cross-ref expansion + reranking."""
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector, NamedSparseVector, SparseVector,
    SearchRequest, FusionQuery, Fusion, PrefetchQuery, Query,
)
from fastembed import SparseTextEmbedding

from .config import Config, cfg as default_cfg
from .models import Chunk, RetrievedChunk


def _e5_query(text: str) -> str:
    return "query: " + text


class Retriever:
    def __init__(self, cfg: Config = default_cfg):
        self.cfg = cfg
        self.client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

        print("Dense model yükleniyor...")
        self.dense = SentenceTransformer(cfg.dense_model)

        print("Sparse model yükleniyor...")
        self.sparse = SparseTextEmbedding(model_name="Qdrant/bm25")

        print("Reranker yükleniyor...")
        self.reranker = CrossEncoder(cfg.reranker_model)

    # ------------------------------------------------------------------ #
    def _dense_vec(self, text: str) -> list[float]:
        return self.dense.encode(_e5_query(text), normalize_embeddings=True).tolist()

    def _sparse_vec(self, text: str) -> SparseVector:
        sv = list(self.sparse.embed([text]))[0]
        return SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())

    # ------------------------------------------------------------------ #
    def _payload_to_chunk(self, payload: dict, score: float, source: str) -> RetrievedChunk:
        atiflar_list = json.loads(payload.get("atiflar_json", "[]"))
        chunk = Chunk(
            id=payload["chunk_id"],
            content=payload.get("raw_text", ""),
            metadata={**payload, "atiflar_list": atiflar_list},
        )
        return RetrievedChunk(chunk=chunk, score=score, source=source)

    # ------------------------------------------------------------------ #
    def hybrid_search(self, query: str) -> list[RetrievedChunk]:
        """RRF fusion: dense + sparse."""
        dense_vec  = self._dense_vec(query)
        sparse_vec = self._sparse_vec(query)

        results = self.client.query_points(
            collection_name=self.cfg.collection_name,
            prefetch=[
                PrefetchQuery(
                    query=NamedVector(name="dense", vector=dense_vec),
                    limit=self.cfg.retrieve_top_k,
                ),
                PrefetchQuery(
                    query=NamedSparseVector(
                        name="sparse",
                        vector=sparse_vec,
                    ),
                    limit=self.cfg.retrieve_top_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.cfg.retrieve_top_k,
            with_payload=True,
        )

        return [
            self._payload_to_chunk(r.payload, r.score, "hybrid")
            for r in results.points
        ]

    # ------------------------------------------------------------------ #
    def expand_refs(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Retrieval sonucu gelen chunk'ların referans ettiği maddeleri çek."""
        existing_ids = {rc.chunk.id for rc in retrieved}
        to_fetch_ids = []

        for rc in retrieved:
            for atif in rc.chunk.atiflar:
                if atif.madde:
                    cid = f"{atif.no}_madde_{atif.madde}"
                else:
                    # Madde belirtilmemiş — kanunun 1. ve 2. maddesi (amaç/kapsam)
                    cid = f"{atif.no}_madde_1"
                if cid not in existing_ids:
                    to_fetch_ids.append(cid)
                    existing_ids.add(cid)

        if not to_fetch_ids:
            return retrieved

        # Hash ID'leri hesapla
        hash_ids = [abs(hash(cid)) % (2**63) for cid in to_fetch_ids]

        results = self.client.retrieve(
            collection_name=self.cfg.collection_name,
            ids=hash_ids,
            with_payload=True,
        )

        extra = [
            self._payload_to_chunk(r.payload, 0.0, "ref_expansion")
            for r in results
            if r.payload
        ]
        return retrieved + extra

    # ------------------------------------------------------------------ #
    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Cross-encoder ile top-K'ya daralt."""
        if not candidates:
            return candidates

        pairs = [(query, rc.chunk.content) for rc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top = ranked[: self.cfg.rerank_top_k]
        for rc, score in top:
            rc.score = float(score)
        return [rc for rc, _ in top]

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Tam pipeline: hybrid → expand → rerank."""
        candidates = self.hybrid_search(query)

        if self.cfg.expand_refs:
            candidates = self.expand_refs(candidates)

        return self.rerank(query, candidates)
