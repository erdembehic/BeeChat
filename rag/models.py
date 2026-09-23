"""Veri modelleri."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Atif:
    no: str
    madde: Optional[str] = None


@dataclass
class Chunk:
    id:       str           # "{mevzuatNo}_madde_{maddeNo}"
    content:  str           # embed edilen metin
    metadata: dict          # Qdrant payload

    @property
    def atiflar(self) -> list[Atif]:
        return [Atif(**a) for a in self.metadata.get("atiflar_list", [])]


@dataclass
class RetrievedChunk:
    chunk:    Chunk
    score:    float
    source:   str           # "vector" | "keyword" | "ref_expansion"


@dataclass
class RAGResult:
    answer:   str
    chunks:   list[RetrievedChunk]
    query:    str

    def sources(self) -> list[str]:
        seen = set()
        out  = []
        for rc in self.chunks:
            label = f"{rc.chunk.metadata['baslik']} Madde {rc.chunk.metadata['maddeNo']}"
            if label not in seen:
                seen.add(label)
                out.append(label)
        return out
