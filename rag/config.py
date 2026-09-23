"""RAG pipeline yapılandırması."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Veri dizinleri
    parsed_itu_dir: Path = field(default_factory=lambda: Path("parsed"))
    parsed_ext_dir: Path = field(default_factory=lambda: Path("parsed_external"))
    atiflar_dir:    Path = field(default_factory=lambda: Path("atiflar"))

    # Qdrant (ortam değişkenleri QDRANT_URL / QDRANT_API_KEY override eder)
    qdrant_host:     str = "localhost"
    qdrant_port:     int = 6333
    collection_name: str = "mevzuat"

    # Embedding (Voyage AI)
    dense_batch_size: int = 128

    # Retrieval
    retrieve_top_k: int  = 20
    rerank_top_k:   int  = 6
    expand_refs:    bool = True

    # Anthropic
    claude_model: str = "claude-sonnet-4-6"
    max_tokens:   int = 2048

    system_prompt: str = (
        "Sen İstanbul Teknik Üniversitesi mevzuat asistanısın. "
        "Yalnızca sağlanan kaynak belgelere dayanarak yanıt ver. "
        "Her iddiayı ilgili yönetmelik/kanun ve madde numarasıyla destekle. "
        "Kaynaklarda cevap yoksa bunu açıkça belirt."
    )


cfg = Config()
