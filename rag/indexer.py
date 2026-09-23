"""Qdrant index oluşturma ve güncelleme."""
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector, NamedSparseVector, NamedVector,
    OptimizersConfigDiff,
)
from fastembed import SparseTextEmbedding

from .config import Config, cfg as default_cfg
from .chunker import load_all_chunks
from .models import Chunk


def _get_client(cfg: Config) -> QdrantClient:
    return QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)


def _ensure_collection(client: QdrantClient, cfg: Config) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if cfg.collection_name in existing:
        print(f"Collection '{cfg.collection_name}' zaten mevcut.")
        return

    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config={
            "dense": VectorParams(size=cfg.dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    print(f"Collection '{cfg.collection_name}' oluşturuldu.")


def build_index(cfg: Config = default_cfg) -> None:
    """Tüm chunk'ları embed edip Qdrant'a yükler."""
    # Modelleri yükle
    print("Dense model yükleniyor:", cfg.dense_model)
    dense_model = SentenceTransformer(cfg.dense_model)

    print("Sparse model yükleniyor (BM25)...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    client = _get_client(cfg)
    _ensure_collection(client, cfg)

    chunks = load_all_chunks(cfg)

    # Batch'ler halinde işle
    batch_size = cfg.dense_batch_size
    for i in tqdm(range(0, len(chunks), batch_size), desc="İndeksleniyor"):
        batch: list[Chunk] = chunks[i : i + batch_size]

        # Dense embedding (e5 prefix zaten content'te)
        contents   = [c.content for c in batch]
        dense_vecs = dense_model.encode(contents, normalize_embeddings=True).tolist()

        # Sparse embedding (BM25) — prefix'siz raw text üzerinde
        raw_texts   = [c.metadata["raw_text"] for c in batch]
        sparse_vecs = list(sparse_model.embed(raw_texts))

        points = []
        for j, chunk in enumerate(batch):
            sv = sparse_vecs[j]
            # Payload: atiflar_list JSON string olarak sakla
            payload = {k: v for k, v in chunk.metadata.items() if k != "atiflar_list"}
            payload["atiflar_json"] = json.dumps(
                chunk.metadata.get("atiflar_list", []), ensure_ascii=False
            )

            points.append(PointStruct(
                id=abs(hash(chunk.id)) % (2**63),
                vector={
                    "dense":  dense_vecs[j],
                    "sparse": SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    ),
                },
                payload={"chunk_id": chunk.id, **payload},
            ))

        client.upsert(collection_name=cfg.collection_name, points=points)

    # İndeksi etkinleştir
    client.update_collection(
        collection_name=cfg.collection_name,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
    )
    print(f"\n✓ {len(chunks)} chunk indekslendi.")


if __name__ == "__main__":
    build_index()
