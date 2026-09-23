#!/usr/bin/env python3
"""
Kullanım:
  # Qdrant'ı Docker'da başlat (bir kez):
  docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

  # İndeksi oluştur:
  python -m rag.run_index
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexer import build_index

if __name__ == "__main__":
    build_index()
