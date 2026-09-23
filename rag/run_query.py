#!/usr/bin/env python3
"""
Kullanım:
  python -m rag.run_query "Doktora öğrencisinin azami öğrenim süresi nedir?"
  python -m rag.run_query "2547 sayılı kanun uyarınca disiplin cezaları" --stream
  python -m rag.run_query "Öğretim üyesi atama şartları" --yonetmelik-only
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("soru", nargs="+")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--yonetmelik-only", action="store_true")
    parser.add_argument("--mevzuat-no", type=str, default=None)
    args = parser.parse_args()

    soru = " ".join(args.soru)
    print(f"\nSoru: {soru}\n{'─'*60}")

    rag = RAGPipeline()

    if args.yonetmelik_only or args.mevzuat_no:
        result = rag.query_with_filter(
            soru,
            sadece_yonetmelik=args.yonetmelik_only,
            mevzuat_no=args.mevzuat_no,
        )
    else:
        result = rag.query(soru, stream=args.stream)

    if not args.stream:
        print(result.answer)

    print(f"\n{'─'*60}")
    print("Kaynaklar:")
    for src in result.sources():
        print(f"  • {src}")


if __name__ == "__main__":
    main()
