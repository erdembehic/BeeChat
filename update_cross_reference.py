#!/usr/bin/env python3
"""
Cross-Reference Haritası Güncelleyici
Dış kanunları da dahil ederek tam çift yönlü atıf haritası oluşturur.
"""

import json
from pathlib import Path
from collections import defaultdict

PARSED_DIR = Path("parsed")
PARSED_EXTERNAL_DIR = Path("parsed_external")
ATIFLAR_DIR = Path("atiflar")


def load_parsed(directory):
    """Bir dizindeki tüm JSON dosyalarını yükler."""
    result = {}
    for pf in sorted(directory.glob("*.json")):
        if pf.name == "index.json":
            continue
        data = json.loads(pf.read_text(encoding="utf-8"))
        no = data["meta"].get("mevzuatNo", pf.stem)
        result[no] = data
    return result


def madde_metni_al(parsed_data, madde_no):
    """Belirtilen madde numarasının metnini döndürür."""
    for bolum in parsed_data.get("bolumler", []):
        for madde in bolum["maddeler"]:
            if madde["no"] == madde_no:
                fikralar = madde.get("fikralar", [])
                metin_parts = []
                for f in fikralar[:3]:  # İlk 3 fıkra
                    metin_parts.append(f["metin"][:300])
                return {
                    "baslik": madde.get("baslik", ""),
                    "metin_ozet": " ".join(metin_parts)[:500]
                }
    return None


def main():
    ATIFLAR_DIR.mkdir(exist_ok=True)

    print("Ayrıştırılmış mevzuatlar yükleniyor...")
    itu_parsed = load_parsed(PARSED_DIR)
    ext_parsed = load_parsed(PARSED_EXTERNAL_DIR)

    print(f"  İTÜ yönetmelikleri: {len(itu_parsed)}")
    print(f"  Dış kanunlar: {len(ext_parsed)}")

    # İç atıf grafiği (İTÜ → İTÜ)
    ic_atif_map = defaultdict(lambda: {
        "kaynak_no": "", "kaynak_baslik": "", "kaynak_tur": "Üniversite Yönetmeliği",
        "hedef_no": "", "hedef_ad": "", "hedef_tur": "Üniversite Yönetmeliği",
        "atif_sayisi": 0, "atiflar": []
    })

    # Dış atıf grafiği (İTÜ → Kanun)
    dis_atif_map = defaultdict(lambda: {
        "kaynak_no": "", "kaynak_baslik": "", "kaynak_tur": "Üniversite Yönetmeliği",
        "hedef_no": "", "hedef_ad": "", "hedef_tur": "Kanun",
        "atif_sayisi": 0,
        "atiflar": []
    })

    for no, data in itu_parsed.items():
        kaynak_baslik = data["meta"].get("baslik", "")
        for bolum in data["bolumler"]:
            for madde in bolum["maddeler"]:
                for atif in madde.get("atiflar", []):
                    tip = atif.get("tip", "")
                    hedef_no = atif.get("no", "")

                    if tip in ("ic_atif", "ic_atif_fikra"):
                        continue  # Kendi maddelerine atıf - burada atla

                    if hedef_no in itu_parsed:
                        # İTÜ → İTÜ
                        key = f"{no}->{hedef_no}"
                        entry = ic_atif_map[key]
                        entry["kaynak_no"] = no
                        entry["kaynak_baslik"] = kaynak_baslik
                        entry["hedef_no"] = hedef_no
                        entry["hedef_ad"] = itu_parsed[hedef_no]["meta"].get("baslik", "")
                        entry["atif_sayisi"] += 1
                        entry["atiflar"].append({
                            "kaynak_madde": madde["no"],
                            "ham": atif.get("ham", "")
                        })

                    elif hedef_no in ext_parsed:
                        # İTÜ → Dış Kanun
                        key = f"{no}->{hedef_no}"
                        entry = dis_atif_map[key]
                        entry["kaynak_no"] = no
                        entry["kaynak_baslik"] = kaynak_baslik
                        entry["hedef_no"] = hedef_no
                        entry["hedef_ad"] = ext_parsed[hedef_no]["meta"].get("baslik", hedef_no)
                        entry["hedef_tur"] = ext_parsed[hedef_no]["meta"].get("tur", "Kanun")
                        entry["atif_sayisi"] += 1

                        ref_madde_str = atif.get("referansMadde", "")
                        ref_madde = int(ref_madde_str) if ref_madde_str else None
                        ref_ozet = None
                        if ref_madde:
                            ref_ozet = madde_metni_al(ext_parsed[hedef_no], ref_madde)

                        entry["atiflar"].append({
                            "kaynak_madde": madde["no"],
                            "hedef_madde": ref_madde,
                            "hedef_madde_ozet": ref_ozet,
                            "ham": atif.get("ham", "")
                        })

    # Sonuçları derle
    ic_list = sorted(ic_atif_map.values(), key=lambda x: -x["atif_sayisi"])
    dis_list = sorted(dis_atif_map.values(), key=lambda x: -x["atif_sayisi"])

    print(f"\nİç atıf ilişkileri (İTÜ↔İTÜ): {len(ic_list)}")
    print(f"Dış atıf ilişkileri (İTÜ→Kanun): {len(dis_list)}")

    # cross_reference_haritasi.json = dış atıf ilişkileri (geriye uyumluluk)
    cr_path = ATIFLAR_DIR / "cross_reference_haritasi.json"
    cr_path.write_text(json.dumps(dis_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {cr_path} güncellendi ({len(dis_list)} dış atıf ilişkisi)")

    # dis_atif_haritasi.json — aynı içerik, açık isimle
    dis_path = ATIFLAR_DIR / "dis_atif_haritasi.json"
    dis_path.write_text(json.dumps(dis_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {dis_path} oluşturuldu ({len(dis_list)} dış atıf ilişkisi)")

    # kapsamli_cross_reference.json — gelecekte iç atıflar eklenirse diye
    kapsamli = ic_list + dis_list
    kapsamli_path = ATIFLAR_DIR / "kapsamli_cross_reference.json"
    kapsamli_path.write_text(json.dumps(kapsamli, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {kapsamli_path} oluşturuldu ({len(kapsamli)} toplam ilişki)")

    # Dış atıf özeti: hangi kanun kaç kez referans alıyor
    dis_hedef_ozet = defaultdict(lambda: {"no": "", "ad": "", "tur": "", "toplam_atif": 0, "kaynak_sayi": 0})
    for entry in dis_list:
        h = entry["hedef_no"]
        dis_hedef_ozet[h]["no"] = h
        dis_hedef_ozet[h]["ad"] = entry["hedef_ad"]
        dis_hedef_ozet[h]["tur"] = entry["hedef_tur"]
        dis_hedef_ozet[h]["toplam_atif"] += entry["atif_sayisi"]
        dis_hedef_ozet[h]["kaynak_sayi"] += 1

    print("\n--- Dış Kanun Atıf Özeti ---")
    for item in sorted(dis_hedef_ozet.values(), key=lambda x: -x["toplam_atif"]):
        print(f"  No:{item['no']:6s}  {item['toplam_atif']:3d} atıf  {item['kaynak_sayi']:2d} yönetmelikten  {item['ad'][:60]}")

    # Dış atıf özet JSON'u
    dis_ozet_path = ATIFLAR_DIR / "dis_atif_ozeti.json"
    dis_ozet_path.write_text(
        json.dumps(sorted(dis_hedef_ozet.values(), key=lambda x: -x["toplam_atif"]), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✓ {dis_ozet_path} oluşturuldu")

    # Global atıf indeksini güncelle (dış atıfları da ekle)
    global_atiflar = []
    for no, data in itu_parsed.items():
        kaynak_baslik = data["meta"].get("baslik", "")
        for bolum in data["bolumler"]:
            for madde in bolum["maddeler"]:
                for atif in madde.get("atiflar", []):
                    hedef_no = atif.get("no", "")
                    if ext_parsed.get(hedef_no):
                        # Zenginleştir: dış kanun bilgisi ekle
                        atif_kopyasi = dict(atif)
                        atif_kopyasi["kaynak_no"] = no
                        atif_kopyasi["kaynak_baslik"] = kaynak_baslik
                        atif_kopyasi["kaynak_madde"] = madde["no"]
                        atif_kopyasi["hedef_tur"] = "dis"
                        atif_kopyasi["hedef_ad_tam"] = ext_parsed[hedef_no]["meta"].get("baslik", "")
                        global_atiflar.append(atif_kopyasi)

    print(f"\nDış kanunlara yapılan toplam atıf: {len(global_atiflar)}")


if __name__ == "__main__":
    main()
