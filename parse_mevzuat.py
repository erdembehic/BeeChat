#!/usr/bin/env python3
"""
İTÜ Mevzuat Parser
Yönetmelikleri madde/fıkra/bent yapısına ayrıştırır ve atıfları çıkarır.
"""

import os
import re
import json
from pathlib import Path

MEVZUAT_DIR = Path("mevzuat")
PARSED_DIR = Path("parsed")
ATIFLAR_DIR = Path("atiflar")

# Türkçe sıra sayıları → rakam
SAYI_MAP = {
    "BİRİNCİ": 1, "İKİNCİ": 2, "ÜÇÜNCÜ": 3, "DÖRDÜNCÜ": 4,
    "BEŞİNCİ": 5, "ALTINCI": 6, "YEDİNCİ": 7, "SEKİZİNCİ": 8,
    "DOKUZUNCU": 9, "ONUNCU": 10, "ON BİRİNCİ": 11, "ON İKİNCİ": 12,
    "ON ÜÇÜNCÜ": 13, "ON DÖRDÜNCÜ": 14, "ON BEŞİNCİ": 15,
    "ON ALTINCI": 16, "ON YEDİNCİ": 17, "ON SEKİZİNCİ": 18,
    "GEÇİCİ": "GEÇİCİ", "SON": "SON"
}

# Atıf kalıpları
ATIF_PATTERNS = [
    # Tarihli ve sayılı - kanun/kararname vb.
    {
        "tip": "tarihli_sayili",
        "re": re.compile(
            r'(\d{1,2}/\d{1,2}/\d{4})\s+tarihli\s+ve\s+(\d+)\s+sayılı\s+'
            r'([\w\s]+?(?:Kanun|Kararname|Yönetmelik|Tebliğ|Tüzük|Karar)(?:u|ü|ı|i|nun|nın|nün|nin|de|da|'
            r'ye|ya|e|a|le|la|den|dan|ten|tan|lar|lerin|lara|lardan|lara)?)',
            re.IGNORECASE
        )
    },
    # Sadece "X sayılı ..."
    {
        "tip": "sayili",
        "re": re.compile(
            r'(?<!\d)(\d+)\s+sayılı\s+'
            r'([\w\s]+?(?:Kanun|Kararname|Yönetmelik|Tebliğ|Tüzük|Karar)(?:u|ü|ı|i|nun|nın|nün|nin|de|da|'
            r'ye|ya|e|a|le|la|den|dan|ten|tan|lar|lerin|lara|lardan|lara)?)',
            re.IGNORECASE
        )
    },
    # Cumhurbaşkanlığı Kararnamesi N/YYYY
    {
        "tip": "cbk",
        "re": re.compile(
            r'(\d+)/(\d{4})\s+sayılı\s+Cumhurbaşkanlığı\s+Kararnamesi',
            re.IGNORECASE
        )
    },
    # Bu Yönetmeliğin X inci/nci/üncü/uncu maddesi - iç atıf
    {
        "tip": "ic_atif",
        "re": re.compile(
            r'[Bb]u\s+Yönetmeliğin?\s+(\d+)\s*(?:inci|nci|üncü|uncu|ıncı|ncı)\s+(?:madde|fıkra)',
            re.IGNORECASE
        )
    },
    # "X inci maddenin Y inci fıkrası" - iç atıf
    {
        "tip": "ic_atif_fikra",
        "re": re.compile(
            r'(\d+)\s*(?:inci|nci|üncü|uncu|ıncı|ncı)\s+maddenin\s+\((\d+)\)\s*(?:inci|nci|üncü|uncu|ıncı|ncı)?\s*fıkra',
            re.IGNORECASE
        )
    },
]

def temizle(s):
    return re.sub(r'\s+', ' ', s).strip()

def parse_header(lines):
    """Başlık bloğunu ayrıştır."""
    meta = {}
    for line in lines[:10]:
        line = line.strip()
        if line.startswith("MEVZUAT NO:"):
            meta["mevzuatNo"] = line.split(":", 1)[1].strip()
        elif line.startswith("RG TARİHİ:"):
            meta["rgTarihi"] = line.split(":", 1)[1].strip()
        elif line.startswith("RG SAYISI:"):
            meta["rgSayisi"] = line.split(":", 1)[1].strip()
        elif line.startswith("KAYNAK:"):
            meta["kaynak"] = line.split(":", 1)[1].strip()
    return meta

def bolum_sirasi(baslik):
    for kelime, sayi in SAYI_MAP.items():
        if kelime in baslik:
            return sayi
    return 99

def atif_cikart(metin, kaynak_madde_no=None):
    """Metinden atıfları çıkart."""
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
                # Madde referansı var mı?
                madde_m = re.search(r'(\d+)\s*(?:inci|nci|üncü|uncu|ıncı|ncı)\s+madde', metin[m.end():m.end()+60])
                if madde_m:
                    atif["referansMadde"] = madde_m.group(1)

            elif pat["tip"] == "sayili":
                atif["no"] = m.group(1)
                atif["ad"] = temizle(m.group(2))

            elif pat["tip"] == "cbk":
                atif["no"] = f"{m.group(1)}/{m.group(2)}"
                atif["ad"] = f"Cumhurbaşkanlığı Kararnamesi {atif['no']}"

            elif pat["tip"] == "ic_atif":
                atif["hedefMadde"] = m.group(1)

            elif pat["tip"] == "ic_atif_fikra":
                atif["hedefMadde"] = m.group(1)
                atif["hedefFikra"] = m.group(2)

            if kaynak_madde_no is not None:
                atif["kaynakMadde"] = kaynak_madde_no

            atiflar.append(atif)

    return atiflar

def parse_file(filepath):
    """Tek bir yönetmelik dosyasını ayrıştır."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    meta = parse_header(lines)

    # Bölüm kalıbı: "BİRİNCİ BÖLÜM" vb. veya "GEÇİCİ MADDE" veya "YÜRÜRLÜK"
    bolum_re = re.compile(
        r'^((?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|'
        r'DOKUZUNCU|ONUNCU|ON\s+\w+İ|GEÇİCİ|EK)\s+BÖLÜM|'
        r'(?:YÜRÜRLÜK|YÜRÜTME)|'
        r'(?:[IVX]+\.\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğışöü\s]+))\s*$'
    )
    # Madde kalıbı (büyük/küçük harf, tire veya em-dash)
    madde_re = re.compile(r'^(?:MADDE|Madde)\s+(\d+)\s*[-–—]\s*(.*)')
    # Geçici madde kalıbı
    gecici_re = re.compile(r'^GEÇİCİ MADDE\s+(\d+)\s*[-–—]\s*(.*)')
    # Fıkra kalıbı
    fikra_re = re.compile(r'^\((\d+)\)\s+(.*)')
    # Bent kalıbı
    bent_re = re.compile(r'^([a-zçğışöü])\)\s+(.*)')

    bolumler = []
    mevcut_bolum = None
    mevcut_madde = None
    mevcut_fikra = None
    baslik_bekle = False  # Bölüm başlığından sonra alt başlık bekliyoruz
    madde_baslik_bekle = False  # Madde satırından önce gelen başlık

    # Başlık (yönetmelik adı) için tarama
    title_lines = []
    header_end = False
    for line in lines:
        stripped = line.strip()
        if "=" * 10 in stripped:
            header_end = True
            continue
        if header_end and stripped:
            title_lines.append(stripped)
        if len(title_lines) >= 5:
            break
    meta["baslik"] = " ".join(l for l in title_lines if l and "YÖNETMELİĞİ" in l or "KARARI" in l or "YÖNETMELİK" in l)
    if not meta["baslik"]:
        meta["baslik"] = " ".join(title_lines[:3])

    pending_baslik = []  # Madde başlığı olabilecek satırlar

    def kaydet_fikra():
        if mevcut_fikra and mevcut_madde:
            mevcut_madde["fikralar"].append(mevcut_fikra)

    def kaydet_madde():
        if mevcut_madde and mevcut_bolum:
            # Madde atıflarını çıkart
            tam_metin = mevcut_madde.get("_ham", "")
            atiflar = atif_cikart(tam_metin, mevcut_madde["no"])
            mevcut_madde["atiflar"] = atiflar
            mevcut_madde.pop("_ham", None)
            mevcut_bolum["maddeler"].append(mevcut_madde)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        i += 1

        if not stripped:
            continue

        # Bölüm başlığı
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

        # Alt başlık (bölüm başlığından sonraki ilk satır)
        if baslik_bekle and mevcut_bolum:
            if not madde_re.match(stripped) and not gecici_re.match(stripped):
                mevcut_bolum["altBaslik"] = stripped
                baslik_bekle = False
                continue
            baslik_bekle = False

        # Madde satırı
        mm = madde_re.match(stripped)
        if not mm:
            mm = gecici_re.match(stripped)
            gecici = True
        else:
            gecici = False

        if mm:
            kaydet_fikra()
            kaydet_madde()
            mevcut_fikra = None

            # Bölüm yoksa varsayılan bölüm oluştur
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

            # Madde başlığı bekleniyorsa al
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

            # Aynı satırda fıkra başlıyorsa
            if devam:
                fm = fikra_re.match(devam)
                if fm:
                    mevcut_fikra = {"no": int(fm.group(1)), "metin": fm.group(2), "bentler": []}
                    mevcut_madde["_ham"] += "\n" + devam
                else:
                    mevcut_fikra = {"no": 1, "metin": devam, "bentler": []}
                    mevcut_madde["_ham"] += "\n" + devam
            continue

        # Fıkra satırı
        fm = fikra_re.match(stripped)
        if fm and mevcut_madde:
            kaydet_fikra()
            mevcut_fikra = {"no": int(fm.group(1)), "metin": temizle(fm.group(2)), "bentler": []}
            if "_ham" in mevcut_madde:
                mevcut_madde["_ham"] += "\n" + stripped
            continue

        # Bent satırı
        bnt = bent_re.match(stripped)
        if bnt and mevcut_fikra:
            mevcut_fikra["bentler"].append({
                "harf": bnt.group(1),
                "metin": temizle(bnt.group(2))
            })
            if mevcut_madde and "_ham" in mevcut_madde:
                mevcut_madde["_ham"] += "\n" + stripped
            continue

        # Devam satırı — mevcut fıkraya ekle
        if mevcut_fikra:
            mevcut_fikra["metin"] += " " + temizle(stripped)
            if mevcut_madde and "_ham" in mevcut_madde:
                mevcut_madde["_ham"] += " " + stripped
            continue

        # Madde başlığı olabilir (madde satırından önce gelir, tek satır, kısa)
        if mevcut_bolum and not mevcut_madde:
            pending_baslik.append(stripped)
        elif mevcut_bolum and mevcut_madde and not mevcut_fikra:
            # Maddenin devam metnini mevcut maddeye ekle
            mevcut_madde["_ham"] = mevcut_madde.get("_ham", "") + " " + stripped
        elif not mevcut_bolum:
            # Henüz bölüm yoksa, madde başlığı tutuyoruz
            pending_baslik.append(stripped)

    # Kalan verileri kaydet
    kaydet_fikra()
    kaydet_madde()

    # Tüm atıfları topla
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
    PARSED_DIR.mkdir(exist_ok=True)
    ATIFLAR_DIR.mkdir(exist_ok=True)

    txt_files = sorted(MEVZUAT_DIR.glob("*.txt"))
    print(f"{len(txt_files)} dosya işlenecek...\n")

    global_atiflar = []  # Tüm atıf ilişkileri
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

            # Parsed JSON kaydet
            parsed_path = PARSED_DIR / f"{no}.json"
            parsed_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            # Atıflar JSON kaydet
            if result["atiflar"]:
                atif_path = ATIFLAR_DIR / f"{no}_atiflar.json"
                atif_path.write_text(
                    json.dumps({
                        "kaynak": {
                            "mevzuatNo": meta.get("mevzuatNo", no),
                            "baslik": meta.get("baslik", ""),
                            "rgTarihi": meta.get("rgTarihi", ""),
                        },
                        "atiflar": result["atiflar"]
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

            # Global atıf listesi için
            for atif in result["atiflar"]:
                global_atiflar.append({
                    "kaynak_no": meta.get("mevzuatNo", no),
                    "kaynak_baslik": meta.get("baslik", ""),
                    **atif
                })

            ozet.append({
                "mevzuatNo": meta.get("mevzuatNo", no),
                "baslik": meta.get("baslik", ""),
                "rgTarihi": meta.get("rgTarihi", ""),
                "bolumSayisi": bolum_sayisi,
                "maddeSayisi": madde_sayisi,
                "atifSayisi": atif_sayisi,
                "parsedDosya": f"parsed/{no}.json"
            })

            print(f"    ✓ {bolum_sayisi} bölüm, {madde_sayisi} madde, {atif_sayisi} atıf")

        except Exception as e:
            print(f"    ✗ HATA: {e}")
            import traceback; traceback.print_exc()

    # Özet index kaydet
    index_path = Path("parsed/index.json")
    index_path.write_text(
        json.dumps(ozet, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Global atıf indeksi kaydet
    global_atif_path = Path("atiflar/global_atif_indeksi.json")
    global_atif_path.write_text(
        json.dumps(global_atiflar, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Cross-reference haritası: hangi mevzuat hangi mevzuata atıf yapıyor
    cr_map = {}
    for atif in global_atiflar:
        src = atif["kaynak_no"]
        hedef_no = atif.get("no", "")
        if hedef_no:
            key = f"{src} → {hedef_no}"
            if key not in cr_map:
                cr_map[key] = {
                    "kaynak_no": src,
                    "kaynak_baslik": atif["kaynak_baslik"],
                    "hedef_no": hedef_no,
                    "hedef_ad": atif.get("ad", ""),
                    "atif_sayisi": 0,
                    "atiflar": []
                }
            cr_map[key]["atif_sayisi"] += 1
            cr_map[key]["atiflar"].append({
                "kaynakMadde": atif.get("kaynakMadde"),
                "ham": atif["ham"]
            })

    cr_list = sorted(cr_map.values(), key=lambda x: x["atif_sayisi"], reverse=True)
    cr_path = Path("atiflar/cross_reference_haritasi.json")
    cr_path.write_text(
        json.dumps(cr_list, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n=== TAMAMLANDI ===")
    print(f"Toplam işlenen: {len(ozet)} yönetmelik")
    print(f"Toplam atıf   : {len(global_atiflar)}")
    print(f"Benzersiz CR  : {len(cr_map)}")
    print(f"\nDosyalar:")
    print(f"  parsed/         → {len(ozet)} adet yapılandırılmış JSON")
    print(f"  atiflar/        → {len(ozet)} adet atıf dosyası + global indeks + CR haritası")

if __name__ == "__main__":
    main()
