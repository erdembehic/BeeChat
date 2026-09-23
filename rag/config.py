"""RAG pipeline yapılandırması."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Veri dizinleri
    parsed_itu_dir:  Path = field(default_factory=lambda: Path("parsed"))
    parsed_ext_dir:  Path = field(default_factory=lambda: Path("parsed_external"))
    atiflar_dir:     Path = field(default_factory=lambda: Path("atiflar"))

    # Qdrant
    qdrant_host:       str  = "localhost"
    qdrant_port:       int  = 6333
    collection_name:   str  = "mevzuat"

    # Embedding — dense
    dense_model:       str  = "intfloat/multilingual-e5-large"
    dense_dim:         int  = 1024
    dense_batch_size:  int  = 32

    # Reranker
    reranker_model:    str  = "BAAI/bge-reranker-v2-m3"

    # Retrieval
    retrieve_top_k:    int  = 20   # dense + sparse'tan ilk N
    rerank_top_k:      int  = 6    # reranker sonrası LLM'e giden
    expand_refs:       bool = True  # cross-ref expansion

    # Anthropic
    claude_model:      str  = "claude-sonnet-4-6"
    max_tokens:        int  = 2048

    # Sistem promptu
    system_prompt: str = (
        "Sen İstanbul Teknik Üniversitesi mevzuat asistanısın. "
        "Yalnızca sağlanan kaynak belgelere dayanarak yanıt ver. "
        "Her iddiayı ilgili yönetmelik/kanun ve madde numarasıyla destekle. "
        "Kaynaklarda cevap yoksa bunu açıkça belirt."
    )


cfg = Config()
