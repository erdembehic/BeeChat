"""JSON → Chunk dönüşümü."""
import json
from pathlib import Path
from .models import Chunk


def _madde_text(madde: dict) -> str:
    """Madde başlık + fıkra + bent metnini tek string'e birleştirir."""
    parts = []
    if madde.get("baslik"):
        parts.append(madde["baslik"])
    for f in madde.get("fikralar", []):
        parts.append(f["metin"])
        for bent in f.get("bentler", []):
            parts.append(f"  {bent['harf']}) {bent['metin']}")
    return "\n".join(parts).strip()


def _e5_format(text: str, is_query: bool = False) -> str:
    """multilingual-e5 passage/query prefix."""
    prefix = "query: " if is_query else "passage: "
    return prefix + text


def load_chunks(parsed_dir: Path, tur: str) -> list[Chunk]:
    chunks = []
    for pf in sorted(parsed_dir.glob("*.json")):
        if pf.name == "index.json":
            continue
        data = json.loads(pf.read_text(encoding="utf-8"))
        meta_base = data["meta"]

        for bolum in data["bolumler"]:
            for madde in bolum["maddeler"]:
                text = _madde_text(madde)
                if not text:
                    continue

                # Referans listesi (sadece dış kanun atıfları)
                atiflar_list = [
                    {"no": a["no"], "madde": a.get("referansMadde")}
                    for a in madde.get("atiflar", [])
                    if a.get("tip") in ("sayili", "tarihli_sayili") and a.get("no")
                ]

                chunk_id = f"{meta_base['mevzuatNo']}_madde_{madde['no']}"

                chunks.append(Chunk(
                    id=chunk_id,
                    content=_e5_format(text),
                    metadata={
                        "mevzuatNo":    meta_base["mevzuatNo"],
                        "baslik":       meta_base["baslik"],
                        "tur":          tur,
                        "maddeNo":      madde["no"],
                        "gecici":       madde.get("gecici", False),
                        "bolumBaslik":  bolum["baslik"],
                        "altBaslik":    bolum.get("altBaslik", ""),
                        "raw_text":     text,       # reranker ve LLM için prefix'siz
                        "atiflar_list": atiflar_list,
                    },
                ))
    return chunks


def load_all_chunks(cfg) -> list[Chunk]:
    itu = load_chunks(cfg.parsed_itu_dir, tur="yonetmelik")
    ext = load_chunks(cfg.parsed_ext_dir, tur="kanun")
    print(f"Chunk: {len(itu)} yönetmelik madde + {len(ext)} kanun madde = {len(itu)+len(ext)} toplam")
    return itu + ext
