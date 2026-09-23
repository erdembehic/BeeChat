"""Dense retrieval + cross-ref expansion + Cohere reranking."""
from __future__ import annotations

import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .config import Config, cfg as default_cfg
from .embedder import embed_query
from .reranker_api import rerank
from .models import Chunk, RetrievedChunk


def _get_client(cfg: Config) -> QdrantClient:
    url     = os.environ.get("QDRANT_URL", f"http://{cfg.qdrant_host}:{cfg.qdrant_port}")
    api_key = os.environ.get("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


class Retriever:
    def __init__(self, cfg: Config = default_cfg):
        self.cfg    = cfg
        self.client = _get_client(cfg)

    # ── Yardımcılar ───────────────────────────────────────────────────── #

    def _payload_to_chunk(self, payload: dict, score: float, source: str) -> RetrievedChunk:
        atiflar_list = json.loads(payload.get("atiflar_json", "[]"))
        chunk = Chunk(
            id=payload["chunk_id"],
            content=payload.get("raw_text", ""),
            metadata={**payload, "atiflar_list": atiflar_list},
        )
        return RetrievedChunk(chunk=chunk, score=score, source=source)

    # ── 1. Dense search ───────────────────────────────────────────────── #

    def dense_search(
        self,
        query: str,
        limit: int | None = None,
        qfilter: Filter | None = None,
    ) -> list[RetrievedChunk]:
        vec = embed_query(query)
        results = self.client.search(
            collection_name=self.cfg.collection_name,
            query_vector=vec,
            limit=limit or self.cfg.retrieve_top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [self._payload_to_chunk(r.payload, r.score, "dense") for r in results]

    # ── 2. Cross-ref expansion ────────────────────────────────────────── #

    def expand_refs(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        existing = {rc.chunk.id for rc in retrieved}
        to_fetch = []
        for rc in retrieved:
            for atif in rc.chunk.atiflar:
                cid = (
                    f"{atif.no}_madde_{atif.madde}"
                    if atif.madde
                    else f"{atif.no}_madde_1"
                )
                if cid not in existing:
                    to_fetch.append(cid)
                    existing.add(cid)

        if not to_fetch:
            return retrieved

        hash_ids = [abs(hash(cid)) % (2**63) for cid in to_fetch]
        records  = self.client.retrieve(
            collection_name=self.cfg.collection_name,
            ids=hash_ids,
            with_payload=True,
        )
        extra = [
            self._payload_to_chunk(r.payload, 0.0, "ref_expansion")
            for r in records
            if r.payload
        ]
        return retrieved + extra

    # ── 3. Rerank ─────────────────────────────────────────────────────── #

    def rerank_chunks(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        docs   = [rc.chunk.content for rc in candidates]
        ranked = rerank(query, docs, top_n=self.cfg.rerank_top_k)
        result = []
        for orig_idx, score in ranked:
            rc       = candidates[orig_idx]
            rc.score = score
            result.append(rc)
        return result

    # ── Ana metod ─────────────────────────────────────────────────────── #

    def retrieve(
        self,
        query: str,
        qfilter: Filter | None = None,
    ) -> list[RetrievedChunk]:
        candidates = self.dense_search(query, qfilter=qfilter)
        if self.cfg.expand_refs:
            candidates = self.expand_refs(candidates)
        return self.rerank_chunks(query, candidates)
