"""Qdrant index oluşturma — Voyage AI dense embedding."""
import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff,
)

from .config import Config, cfg as default_cfg
from .chunker import load_all_chunks
from .embedder import embed_documents, DIM
from .models import Chunk


def _get_client(cfg: Config) -> QdrantClient:
    import os
    url    = os.environ.get("QDRANT_URL",    f"http://{cfg.qdrant_host}:{cfg.qdrant_port}")
    api_key = os.environ.get("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(client: QdrantClient, cfg: Config) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if cfg.collection_name in existing:
        print(f"Collection '{cfg.collection_name}' zaten mevcut.")
        return
    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    print(f"Collection '{cfg.collection_name}' oluşturuldu.")


def build_index(cfg: Config = default_cfg) -> None:
    client = _get_client(cfg)
    _ensure_collection(client, cfg)
    chunks = load_all_chunks(cfg)

    batch_size = cfg.dense_batch_size
    for i in tqdm(range(0, len(chunks), batch_size), desc="İndeksleniyor"):
        batch: list[Chunk] = chunks[i : i + batch_size]
        raw_texts = [c.metadata["raw_text"] for c in batch]
        dense_vecs = embed_documents(raw_texts)

        points = []
        for j, chunk in enumerate(batch):
            payload = {k: v for k, v in chunk.metadata.items() if k != "atiflar_list"}
            payload["atiflar_json"] = json.dumps(
                chunk.metadata.get("atiflar_list", []), ensure_ascii=False
            )
            points.append(PointStruct(
                id=abs(hash(chunk.id)) % (2**63),
                vector=dense_vecs[j],
                payload={"chunk_id": chunk.id, **payload},
            ))
        client.upsert(collection_name=cfg.collection_name, points=points)

    client.update_collection(
        collection_name=cfg.collection_name,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
    )
    print(f"\n✓ {len(chunks)} chunk indekslendi.")


if __name__ == "__main__":
    build_index()
