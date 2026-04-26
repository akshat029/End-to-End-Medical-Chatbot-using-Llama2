from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import settings
from src.data_loader import chunk_text


MATRIX_FILE = "tfidf_matrix.npy"
META_FILE = "metadata.json"
VECTORIZER_FILE = "vectorizer.pkl"


class LocalVectorStore:
    def __init__(self, store_dir: str | None = None):
        self.store_dir = Path(store_dir or settings.vector_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix: np.ndarray | None = None
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

        matrix = self.vectorizer.fit_transform(texts).astype(np.float32)
        self.matrix = matrix.toarray()
        self.metadata = metadata

    def save(self) -> None:
        if self.matrix is None:
            raise RuntimeError("Vector matrix is not built.")

        np.save(str(self.store_dir / MATRIX_FILE), self.matrix)
        (self.store_dir / META_FILE).write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")
        with (self.store_dir / VECTORIZER_FILE).open("wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self) -> None:
        matrix_path = self.store_dir / MATRIX_FILE
        meta_path = self.store_dir / META_FILE
        vectorizer_path = self.store_dir / VECTORIZER_FILE
        if not all(path.exists() for path in [matrix_path, meta_path, vectorizer_path]):
            raise FileNotFoundError(
                f"Missing vector store files in {self.store_dir}. Run ingest.py first."
            )

        self.matrix = np.load(str(matrix_path))
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        with vectorizer_path.open("rb") as f:
            self.vectorizer = pickle.load(f)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, str]]:
        if self.matrix is None:
            raise RuntimeError("Vector matrix is not loaded.")

        k = top_k or settings.top_k
        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)

        doc_norms = np.linalg.norm(self.matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-8
        scores = (self.matrix @ query_vec.T).ravel() / denom

        best_indices = np.argsort(scores)[::-1][:k]
        hits: list[dict[str, str]] = []

        for idx in best_indices:
            if idx < 0 or idx >= len(self.metadata):
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = score
            hits.append(item)
        return hits
