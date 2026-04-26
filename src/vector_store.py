from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.data_loader import chunk_text


INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"


class LocalVectorStore:
    def __init__(self, store_dir: str | None = None, embedding_model_id: str | None = None):
        self.store_dir = Path(store_dir or settings.vector_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_id = embedding_model_id or settings.embedding_model_id
        self.embedder = SentenceTransformer(self.embedding_model_id)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, str]] = []

    def build(self, documents: list[dict[str, str]], chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        texts: list[str] = []
        metadata: list[dict[str, str]] = []

        for doc in documents:
            for chunk in chunk_text(doc["content"], chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                chunk = chunk.strip()
                if not chunk:
                    continue
                texts.append(chunk)
                metadata.append({"source": doc["source"], "text": chunk})

        if not texts:
            raise ValueError("No chunks produced from input documents.")

        vectors = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index = index
        self.metadata = metadata

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Vector index is not built.")

        faiss.write_index(self.index, str(self.store_dir / INDEX_FILE))
        (self.store_dir / META_FILE).write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        index_path = self.store_dir / INDEX_FILE
        meta_path = self.store_dir / META_FILE

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing vector store files in {self.store_dir}. Run ingest.py first."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, str]]:
        if self.index is None:
            raise RuntimeError("Vector index is not loaded.")

        k = top_k or settings.top_k
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, k)
        hits: list[dict[str, str]] = []

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            hits.append(item)
        return hits
