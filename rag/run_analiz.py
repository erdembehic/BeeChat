#!/usr/bin/env python3
"""
Kullanım:
  python -m rag.run_analiz
  python -m rag.run_analiz "Doktora 4. yılındayım, GNO 2.05, tez önerim reddedildi"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.situation import DurumAnalizoru


ORNEKLER = [
    "Lisans 4. sınıfım, GNO'm 1.78, bu dönem 2 dersimi daha geçemedim. Ne olur?",
    "Yüksek lisans 2. yılındayım, tez danışmanım istifa etti, ne yapmalıyım?",
    "Doktora 5. yılındayım, tez savunmam reddedildi. Kaç hakkım var?",
]


def main():
    if len(sys.argv) > 1:
        durum = " ".join(sys.argv[1:])
    else:
        print("Durum analizi — örnek senaryolar:")
        for i, o in enumerate(ORNEKLER, 1):
            print(f"  {i}. {o}")
        print()
        secim = input("Seçim (1-3) veya kendi durumunuzu yazın: ").strip()

        if secim in ("1", "2", "3"):
            durum = ORNEKLER[int(secim) - 1]
        else:
            durum = secim

    print(f"\n{'═'*60}")
    print(f"Durum: {durum}")
    print(f"{'═'*60}\n")

    analizor = DurumAnalizoru()
    sonuc = analizor.analiz_et(durum, stream=True)

    print(f"\n{'─'*60}")
    print("Kullanılan kaynaklar:")
    seen = set()
    for rc in sonuc.kaynaklar:
        m   = rc.chunk.metadata
        lbl = f"{m['baslik']} — Madde {m['maddeNo']}"
        if lbl not in seen:
            seen.add(lbl)
            tag = " (çapraz ref)" if rc.source == "ref_expansion" else ""
            print(f"  • {lbl}{tag}")


if __name__ == "__main__":
    main()
