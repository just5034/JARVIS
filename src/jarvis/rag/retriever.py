"""FAISS-based retrieval for physics knowledge passages."""

from __future__ import annotations

from pathlib import Path


class PhysicsRetriever:
    """Retrieves relevant knowledge passages from the FAISS index."""

    def __init__(self, index_path: Path | None = None) -> None:
        self._index = None
        self._index_path = index_path

    def load(self) -> None:
        raise NotImplementedError("Phase 6: load FAISS index and embedding model")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        raise NotImplementedError("Phase 6: embed query and retrieve passages")
