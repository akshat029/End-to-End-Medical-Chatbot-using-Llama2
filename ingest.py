from __future__ import annotations

import argparse

from src.config import settings
from src.data_loader import load_documents
from src.vector_store import LocalVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local vector index from medical documents")
    parser.add_argument("--data-dir", default=settings.data_dir, help="Directory with source documents")
    parser.add_argument("--store-dir", default=settings.vector_store_dir, help="Directory to store index")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_documents(args.data_dir)

    store = LocalVectorStore(store_dir=args.store_dir)
    store.build(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    store.save()

    print(f"Indexed {len(docs)} documents into {args.store_dir}")


if __name__ == "__main__":
    main()
