"""
Öğrenci durum analizi.
Serbest metin → fact extraction → targeted retrieval → kişiselleştirilmiş analiz.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .config import Config, cfg as default_cfg
from . import llm
from .retriever import Retriever
from .models import RetrievedChunk


# ── Veri modelleri ────────────────────────────────────────────────────────── #

@dataclass
class OgrenciFaktleri:
    program:        Optional[str]  = None   # lisans / yüksek lisans / doktora
    yil:            Optional[int]  = None   # kaçıncı yıl
    donem:          Optional[int]  = None   # güz / bahar
    gno:            Optional[float] = None  # not ortalaması
    basarisiz_ders: Optional[int]  = None   # başarısız ders sayısı
    eksik_kredi:    Optional[int]  = None   # alınması gereken kalan kredi
    izin_durumu:    Optional[str]  = None   # kayıt dondurma, izinli, vb.
    disiplin:       Optional[bool] = None   # disiplin süreci var mı
    tez_durumu:     Optional[str]  = None   # tez önerisi, savunma, vb.
    diger:          list[str]       = field(default_factory=list)
    ham_giris:      str             = ""    # orijinal metin


@dataclass
class DurumAnalizi:
    ogrenci:      OgrenciFaktleri
    ozet:         str
    gecerli_maddeler: list[str]     # ["2547 Madde 44", "İTÜ Lisans Yönetmeliği Madde 12"]
    riskler:      list[str]
    yapabilecekler: list[str]
    adimlar:      list[str]
    sure_ve_tarihler: list[str]
    kaynaklar:    list[RetrievedChunk]
    ham_analiz:   str


# ── Yardımcı sabitler ─────────────────────────────────────────────────────── #

_FACT_EXTRACTION_PROMPT = """\
Aşağıdaki öğrenci ifadesinden yapılandırılmış bilgileri çıkar.
Belirtilmemiş alanları null bırak. Sadece JSON döndür, açıklama ekleme.

İfade:
{metin}

JSON şeması:
{{
  "program": "lisans|yüksek lisans|doktora|null",
  "yil": <int veya null>,
  "donem": <int veya null>,
  "gno": <float veya null>,
  "basarisiz_ders": <int veya null>,
  "eksik_kredi": <int veya null>,
  "izin_durumu": "<string veya null>",
  "disiplin": <true|false|null>,
  "tez_durumu": "<string veya null>",
  "diger": ["<diğer önemli bilgiler>"]
}}"""

_ANALYSIS_SYSTEM = """\
Sen İstanbul Teknik Üniversitesi mevzuat uzmanısın.
Öğrencinin durumunu verilen yönetmelik ve kanun maddelerine göre analiz et.
Yalnızca sağlanan kaynak belgelerden bilgi kullan.
Belirsizlik varsa belirt, tahmin yapma."""

_ANALYSIS_USER = """\
## Öğrenci Durumu
{ham_giris}

## Çıkarılan Bilgiler
{faktler}

## İlgili Mevzuat
{kaynaklar}

---

Lütfen şu başlıklar altında analiz yap:

### Durum Özeti
(Öğrencinin mevcut durumunu 2-3 cümleyle özetle)

### Geçerli Mevzuat Maddeleri
(Hangi yönetmelik/kanun/madde uygulanır — her biri için kısa açıklama)

### Riskler
(Bu durumun olası olumsuz sonuçları — maddeye dayandırarak)

### Yapabilecekler
(Öğrencinin haklarını ve seçeneklerini listele)

### Önerilen Adımlar
(Somut, öncelik sıralı adımlar)

### Süreler ve Tarihler
(Varsa başvuru süreleri, itiraz pencereleri, azami süreler)"""


# ── Ana sınıf ─────────────────────────────────────────────────────────────── #

class DurumAnalizoru:
    def __init__(self, cfg: Config = default_cfg):
        self.cfg = cfg
        self.retriever = Retriever(cfg)
        pass  # llm modülü üzerinden çalışır

    # ── 1. Fact extraction ─────────────────────────────────────────────── #

    def _fact_extraction(self, metin: str) -> OgrenciFaktleri:
        response_text = llm.complete(
            system="",
            user=_FACT_EXTRACTION_PROMPT.format(metin=metin),
            cfg_model=self.cfg.claude_model,
            max_tokens=512,
        )
        # Geçici wrapper — aşağıdaki json parse için
        class _R:
            def __init__(self, t): self.content = [type("C", (), {"text": t})()]
        response = _R(response_text)
        raw = response_text.strip()
        # JSON bloğu içindeyse çıkar
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        return OgrenciFaktleri(
            program=data.get("program"),
            yil=data.get("yil"),
            donem=data.get("donem"),
            gno=data.get("gno"),
            basarisiz_ders=data.get("basarisiz_ders"),
            eksik_kredi=data.get("eksik_kredi"),
            izin_durumu=data.get("izin_durumu"),
            disiplin=data.get("disiplin"),
            tez_durumu=data.get("tez_durumu"),
            diger=data.get("diger", []),
            ham_giris=metin,
        )

    # ── 2. Targeted query generation ──────────────────────────────────── #

    def _queries_from_facts(self, faktler: OgrenciFaktleri) -> list[str]:
        """Fact'lara göre 3-5 hedefli retrieval sorgusu üret."""
        queries = []

        # Program ve yıl bazlı sorgu
        if faktler.program:
            queries.append(f"{faktler.program} azami öğrenim süresi")
            queries.append(f"{faktler.program} kayıt silme şartları")

        # GNO sorgusu
        if faktler.gno is not None:
            queries.append(f"genel not ortalaması başarı şartı {faktler.program or ''}")

        # Başarısız ders
        if faktler.basarisiz_ders:
            queries.append("başarısız ders tekrar kayıt şartları")
            queries.append("ders tekrarı sınav hakkı")

        # İzin/kayıt dondurma
        if faktler.izin_durumu:
            queries.append(f"kayıt dondurma izin {faktler.izin_durumu}")

        # Tez
        if faktler.tez_durumu:
            queries.append(f"tez {faktler.tez_durumu} süresi şartlar")

        # Disiplin
        if faktler.disiplin:
            queries.append("disiplin cezası öğrenci kayıt silme")

        # Diğer notlar
        for d in faktler.diger:
            queries.append(d[:80])

        # Ham metin de doğrudan sorgula
        queries.append(faktler.ham_giris[:120])

        return queries[:6]  # En fazla 6 sorgu

    # ── 3. Multi-query retrieval ───────────────────────────────────────── #

    def _multi_retrieve(self, queries: list[str]) -> list[RetrievedChunk]:
        seen_ids: set[str] = set()
        all_chunks: list[RetrievedChunk] = []

        for q in queries:
            chunks = self.retriever.retrieve(q)
            for rc in chunks:
                if rc.chunk.id not in seen_ids:
                    seen_ids.add(rc.chunk.id)
                    all_chunks.append(rc)

        # Skora göre sırala, en iyi 10'u al
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        return all_chunks[:10]

    # ── 4. Analiz üretimi ─────────────────────────────────────────────── #

    def _build_source_text(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for rc in chunks:
            m = rc.chunk.metadata
            tag = "(çapraz referans)" if rc.source == "ref_expansion" else ""
            parts.append(
                f"[{m['baslik']} — Madde {m['maddeNo']}] {tag}\n{rc.chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _faktler_str(self, f: OgrenciFaktleri) -> str:
        rows = []
        if f.program:        rows.append(f"Program: {f.program}")
        if f.yil:            rows.append(f"Yıl: {f.yil}")
        if f.donem:          rows.append(f"Dönem: {f.donem}")
        if f.gno is not None: rows.append(f"GNO: {f.gno}")
        if f.basarisiz_ders: rows.append(f"Başarısız ders: {f.basarisiz_ders}")
        if f.eksik_kredi:    rows.append(f"Eksik kredi: {f.eksik_kredi}")
        if f.izin_durumu:    rows.append(f"İzin/kayıt durumu: {f.izin_durumu}")
        if f.disiplin:       rows.append("Disiplin süreci: evet")
        if f.tez_durumu:     rows.append(f"Tez durumu: {f.tez_durumu}")
        for d in f.diger:    rows.append(f"Diğer: {d}")
        return "\n".join(rows) if rows else "(bilgi çıkarılamadı)"

    # ── Ana metod ─────────────────────────────────────────────────────── #

    def analiz_et(self, durum: str, stream: bool = False) -> DurumAnalizi:
        print("Durum analiz ediliyor...")

        # Adım 1: Fact extraction
        print("  ► Bilgiler çıkarılıyor...")
        faktler = self._fact_extraction(durum)

        # Adım 2: Hedefli sorgular
        queries = self._queries_from_facts(faktler)
        print(f"  ► {len(queries)} hedefli sorgu oluşturuldu")

        # Adım 3: Multi-query retrieval
        print("  ► Mevzuat aranıyor...")
        kaynaklar = self._multi_retrieve(queries)
        print(f"  ► {len(kaynaklar)} ilgili madde bulundu")

        # Adım 4: Analiz
        print("  ► Analiz üretiliyor...\n")
        user_msg = _ANALYSIS_USER.format(
            ham_giris=durum,
            faktler=self._faktler_str(faktler),
            kaynaklar=self._build_source_text(kaynaklar),
        )

        if stream:
            parts = []
            for text in llm.stream(_ANALYSIS_SYSTEM, user_msg,
                                   self.cfg.claude_model, self.cfg.max_tokens):
                print(text, end="", flush=True)
                parts.append(text)
            print()
            ham_analiz = "".join(parts)
        else:
            ham_analiz = llm.complete(_ANALYSIS_SYSTEM, user_msg,
                                      self.cfg.claude_model, self.cfg.max_tokens)

        # Basit bölüm ayrıştırma
        def _bolum(baslik: str) -> str:
            marker = f"### {baslik}"
            if marker not in ham_analiz:
                return ""
            sonraki = ham_analiz.find("### ", ham_analiz.index(marker) + len(marker))
            return ham_analiz[ham_analiz.index(marker) + len(marker):
                              sonraki if sonraki != -1 else None].strip()

        def _liste(baslik: str) -> list[str]:
            bolum = _bolum(baslik)
            return [
                line.lstrip("-•* ").strip()
                for line in bolum.splitlines()
                if line.strip() and not line.startswith("#")
            ]

        return DurumAnalizi(
            ogrenci=faktler,
            ozet=_bolum("Durum Özeti"),
            gecerli_maddeler=_liste("Geçerli Mevzuat Maddeleri"),
            riskler=_liste("Riskler"),
            yapabilecekler=_liste("Yapabilecekler"),
            adimlar=_liste("Önerilen Adımlar"),
            sure_ve_tarihler=_liste("Süreler ve Tarihler"),
            kaynaklar=kaynaklar,
            ham_analiz=ham_analiz,
        )
