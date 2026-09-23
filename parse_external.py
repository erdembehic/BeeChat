#!/usr/bin/env python3
"""
Dış Kanun Parser
İTÜ yönetmeliklerinde atıf yapılan kanunları madde/fıkra/bent yapısına ayrıştırır.
"""

import os
import re
import json
from pathlib import Path

EXTERNAL_DIR = Path("external")
PARSED_EXTERNAL_DIR = Path("parsed_external")
ATIFLAR_DIR = Path("atiflar")

SAYI_MAP = {
    "BİRİNCİ": 1, "İKİNCİ": 2, "ÜÇÜNCÜ": 3, "DÖRDÜNCÜ": 4,
    "BEŞİNCİ": 5, "ALTINCI": 6, "YEDİNCİ": 7, "SEKİZİNCİ": 8,
    "DOKUZUNCU": 9, "ONUNCU": 10, "ON BİRİNCİ": 11, "ON İKİNCİ": 12,
    "ON ÜÇÜNCÜ": 13, "ON DÖRDÜNCÜ": 14, "ON BEŞİNCİ": 15,
    "ON ALTINCI": 16, "ON YEDİNCİ": 17, "ON SEKİZİNCİ": 18,
    "GEÇİCİ": "GEÇİCİ", "EK": "EK", "SON": "SON"
}

ATIF_PATTERNS = [
    {
        "tip": "tarihli_sayili",
        "re": re.compile(
            r'(\d{1,2}/\d{1,2}/\d{4})\s+tarihli\s+ve\s+(\d+)\s+sayılı\s+'
            r'([\w\s]+?(?:Kanun|Kararname|Yönetmelik|Tebliğ|Tüzük|Karar)(?:u|ü|ı|i|nun|nın|nün|nin|de|da|'
            r'ye|ya|e|a|le|la|den|dan|ten|tan|lar|lerin|lara|lardan|lara)?)',
            re.IGNORECASE
        )
    },
    {
        "tip": "sayili",
        "re": re.compile(
            r'(?<!\d)(\d+)\s+sayılı\s+'
            r'([\w\s]+?(?:Kanun|Kararname|Yönetmelik|Tebliğ|Tüzük|Karar)(?:u|ü|ı|i|nun|nın|nün|nin|de|da|'
            r'ye|ya|e|a|le|la|den|dan|ten|tan|lar|lerin|lara|lardan|lara)?)',
            re.IGNORECASE
        )
    },
    {
        "tip": "cbk",
        "re": re.compile(
            r'(\d+)/(\d{4})\s+sayılı\s+Cumhurbaşkanlığı\s+Kararnamesi',
            re.IGNORECASE
        )
    },
    {
        "tip": "ic_atif",
        "re": re.compile(
            r'[Bb]u\s+Kanun(?:un)?\s+(\d+)\s*(?:inci|nci|üncü|uncu|ıncı|ncı)\s+(?:madde|fıkra)',
            re.IGNORECASE
        )
    },
]


def temizle(s):
    return re.sub(r'\s+', ' ', s).strip()


def bolum_sirasi(baslik):
    for kelime, sayi in SAYI_MAP.items():
        if kelime in baslik:
            return sayi
    return 99


def atif_cikart(metin, kaynak_madde_no=None):
    atiflar = []
    seen = set()
    for pat in ATIF_PATTERNS:
        for m in pat["re"].finditer(metin):
            ham = temizle(m.group(0))
            if ham in seen:
                continue
            seen.add(ham)
            atif = {"tip": pat["tip"], "ham": ham}
            if pat["tip"] == "tarihli_sayili":
                atif["tarih"] = m.group(1)
                atif["no"] = m.group(2)
                atif["ad"] = temizle(m.group(3))
            elif pat["tip"] == "sayili":
                atif["no"] = m.group(1)
                atif["ad"] = temizle(m.group(2))
            elif pat["tip"] == "cbk":
                atif["no"] = f"{m.group(1)}/{m.group(2)}"
                atif["ad"] = f"Cumhurbaşkanlığı Kararnamesi {atif['no']}"
            elif pat["tip"] == "ic_atif":
                atif["hedefMadde"] = m.group(1)
            if kaynak_madde_no is not None:
                atif["kaynakMadde"] = kaynak_madde_no
            atiflar.append(atif)
    return atiflar


def parse_external_header(lines):
    """Dış kanun dosyasının başlık bloğunu ayrıştır."""
    meta = {}
    for line in lines[:10]:
        stripped = line.strip()
        if stripped.startswith("MEVZUAT NO:"):
            meta["mevzuatNo"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("AD:"):
            meta["baslik"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("KABUL TARİHİ:"):
            meta["kabulTarihi"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("MEVZUAT TUR:"):
            meta["tur"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("KAYNAK:"):
            meta["kaynak"] = stripped.split(":", 1)[1].strip()
    return meta


def parse_file(filepath):
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    meta = parse_external_header(lines)

    bolum_re = re.compile(
        r'^((?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|'
        r'DOKUZUNCU|ONUNCU|ON\s+\w+İ|GEÇİCİ|EK)\s+BÖLÜM|'
        r'(?:YÜRÜRLÜK|YÜRÜTME)|'
        r'(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s+KISIM|'
        r'(?:[IVX]+\.\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğışöü\s]+))\s*$'
    )
    madde_re = re.compile(r'^(?:MADDE|Madde)\s+(\d+)\s*[-–—]\s*(.*)')
    gecici_re = re.compile(r'^(?:GEÇİCİ MADDE|Geçici Madde)\s+(\d+)\s*[-–—]\s*(.*)')
    fikra_re = re.compile(r'^\((\d+)\)\s+(.*)')
    bent_re = re.compile(r'^([a-zçğışöü])\)\s+(.*)')

    bolumler = []
    mevcut_bolum = None
    mevcut_madde = None
    mevcut_fikra = None
    baslik_bekle = False
    pending_baslik = []

    def kaydet_fikra():
        if mevcut_fikra and mevcut_madde:
            mevcut_madde["fikralar"].append(mevcut_fikra)

    def kaydet_madde():
        if mevcut_madde and mevcut_bolum:
            tam_metin = mevcut_madde.get("_ham", "")
            atiflar = atif_cikart(tam_metin, mevcut_madde["no"])
            mevcut_madde["atiflar"] = atiflar
            mevcut_madde.pop("_ham", None)
            mevcut_bolum["maddeler"].append(mevcut_madde)

    header_passed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        i += 1

        if not header_passed:
            if "=" * 10 in stripped:
                header_passed = True
            continue

        if not stripped:
            continue

        bm = bolum_re.match(stripped)
        if bm:
            kaydet_fikra()
            kaydet_madde()
            mevcut_fikra = None
            mevcut_madde = None
            pending_baslik = []
            mevcut_bolum = {
                "sira": bolum_sirasi(stripped),
                "baslik": stripped,
                "altBaslik": "",
                "maddeler": []
            }
            bolumler.append(mevcut_bolum)
            baslik_bekle = True
            continue

        if baslik_bekle and mevcut_bolum:
            if not madde_re.match(stripped) and not gecici_re.match(stripped):
                mevcut_bolum["altBaslik"] = stripped
                baslik_bekle = False
                continue
            baslik_bekle = False

        mm = madde_re.match(stripped)
        if not mm:
            mm2 = gecici_re.match(stripped)
            gecici = True if mm2 else False
            mm = mm2
        else:
            gecici = False

        if mm:
            kaydet_fikra()
            kaydet_madde()
            mevcut_fikra = None

            if mevcut_bolum is None:
                mevcut_bolum = {
                    "sira": 1,
                    "baslik": "GENEL",
                    "altBaslik": "",
                    "maddeler": []
                }
                bolumler.append(mevcut_bolum)

            madde_no = int(mm.group(1))
            devam = temizle(mm.group(2))
            madde_baslik = " ".join(pending_baslik).strip() if pending_baslik else ""
            pending_baslik = []

            mevcut_madde = {
                "no": madde_no,
                "gecici": gecici,
                "baslik": madde_baslik,
                "fikralar": [],
                "_ham": devam,
                "atiflar": []
            }

            if devam:
                fm = fikra_re.match(devam)
                if fm:
                    mevcut_fikra = {"no": int(fm.group(1)), "metin": fm.group(2), "bentler": []}
                    mevcut_madde["_ham"] += "\n" + devam
                else:
                    mevcut_fikra = {"no": 1, "metin": devam, "bentler": []}
                    mevcut_madde["_ham"] += "\n" + devam
            continue

        fm = fikra_re.match(stripped)
        if fm and mevcut_madde:
            kaydet_fikra()
            mevcut_fikra = {"no": int(fm.group(1)), "metin": temizle(fm.group(2)), "bentler": []}
            if "_ham" in mevcut_madde:
                mevcut_madde["_ham"] += "\n" + stripped
            continue

        bnt = bent_re.match(stripped)
        if bnt and mevcut_fikra:
            mevcut_fikra["bentler"].append({"harf": bnt.group(1), "metin": temizle(bnt.group(2))})
            if mevcut_madde and "_ham" in mevcut_madde:
                mevcut_madde["_ham"] += "\n" + stripped
            continue

        if mevcut_fikra:
            mevcut_fikra["metin"] += " " + temizle(stripped)
            if mevcut_madde and "_ham" in mevcut_madde:
                mevcut_madde["_ham"] += " " + stripped
            continue

        if mevcut_bolum and not mevcut_madde:
            pending_baslik.append(stripped)
        elif mevcut_bolum and mevcut_madde and not mevcut_fikra:
            mevcut_madde["_ham"] = mevcut_madde.get("_ham", "") + " " + stripped
        elif not mevcut_bolum:
            pending_baslik.append(stripped)

    kaydet_fikra()
    kaydet_madde()

    tum_atiflar = []
    for bolum in bolumler:
        for madde in bolum["maddeler"]:
            tum_atiflar.extend(madde.get("atiflar", []))

    return {
        "meta": meta,
        "bolumler": bolumler,
        "atiflar": tum_atiflar
    }


def main():
    PARSED_EXTERNAL_DIR.mkdir(exist_ok=True)
    ATIFLAR_DIR.mkdir(exist_ok=True)

    txt_files = sorted(EXTERNAL_DIR.glob("*.txt"))
    print(f"{len(txt_files)} dış kanun dosyası işlenecek...\n")

    ozet = []

    for filepath in txt_files:
        no = filepath.stem.split("_")[0]
        print(f"  Ayrıştırılıyor: {no} - {filepath.name[:60]}...")

        try:
            result = parse_file(filepath)
            meta = result["meta"]
            bolum_sayisi = len(result["bolumler"])
            madde_sayisi = sum(len(b["maddeler"]) for b in result["bolumler"])
            atif_sayisi = len(result["atiflar"])

            parsed_path = PARSED_EXTERNAL_DIR / f"{no}.json"
            parsed_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            print(f"    ✓ {bolum_sayisi} bölüm, {madde_sayisi} madde, {atif_sayisi} atıf")

            ozet.append({
                "mevzuatNo": no,
                "baslik": meta.get("baslik", ""),
                "tur": meta.get("tur", "Kanun"),
                "kabulTarihi": meta.get("kabulTarihi", ""),
                "bolumSayisi": bolum_sayisi,
                "maddeSayisi": madde_sayisi,
                "atifSayisi": atif_sayisi
            })

        except Exception as e:
            print(f"    ✗ HATA: {e}")

    index_path = PARSED_EXTERNAL_DIR / "index.json"
    index_path.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== TAMAMLANDI: {len(ozet)} kanun ayrıştırıldı ===")
    print(f"Özet: {PARSED_EXTERNAL_DIR}/index.json")


if __name__ == "__main__":
    main()
