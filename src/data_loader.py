from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: str) -> list[dict[str, str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    docs: list[dict[str, str]] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = _read_pdf(file_path) if file_path.suffix.lower() == ".pdf" else _read_text(file_path)
        content = content.strip()
        if not content:
            continue

        docs.append({"source": str(file_path), "content": content})

    if not docs:
        raise ValueError(
            f"No supported documents found in {root}. Add .txt, .md, or .pdf files."
        )
    return docs


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> Iterable[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield text[start:end]
        if end == n:
            break
        start = end - chunk_overlap
