"""FAISS-based retrieval for physics knowledge passages.

Supports two modes:
1. Full mode: sentence-transformers + FAISS for semantic search
2. Fallback mode: simple keyword matching (no GPU/extra deps needed)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PhysicsRetriever:
    """Retrieves relevant knowledge passages from a corpus."""

    def __init__(self, corpus_path: Path | None = None, index_path: Path | None = None) -> None:
        self._corpus: list[dict] = []
        self._corpus_path = corpus_path
        self._index_path = index_path
        self._index: Any = None
        self._embedding_model: Any = None
        self._use_faiss = False
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def load(self) -> None:
        """Load the corpus and optionally the FAISS index."""
        if self._corpus_path and self._corpus_path.exists():
            with open(self._corpus_path, encoding="utf-8") as f:
                self._corpus = json.load(f)
            logger.info("Loaded %d passages from %s", len(self._corpus), self._corpus_path)
        else:
            logger.warning("No corpus found at %s", self._corpus_path)
            return

        # Try to load FAISS index and embedding model
        if self._index_path and self._index_path.exists():
            try:
                import faiss
                import numpy as np

                self._index = faiss.read_index(str(self._index_path))
                self._use_faiss = True
                logger.info("Loaded FAISS index from %s", self._index_path)
            except ImportError:
                logger.info("faiss-cpu not installed, using keyword retrieval")
            except Exception as e:
                logger.warning("Failed to load FAISS index: %s, using keyword retrieval", e)

        # Try to load embedding model for query embedding
        if self._use_faiss:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded embedding model for semantic search")
            except ImportError:
                logger.info("sentence-transformers not installed, using keyword retrieval")
                self._use_faiss = False

        self._loaded = True

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve the most relevant passages for a query."""
        if not self._loaded or not self._corpus:
            return []

        if self._use_faiss and self._index is not None and self._embedding_model is not None:
            return self._retrieve_faiss(query, top_k)
        return self._retrieve_keywords(query, top_k)

    def _retrieve_faiss(self, query: str, top_k: int) -> list[str]:
        """Semantic retrieval using FAISS."""
        import numpy as np

        query_embedding = self._embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(top_k, len(self._corpus))
        distances, indices = self._index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._corpus):
                results.append(self._corpus[idx]["text"])
        return results

    def _retrieve_keywords(self, query: str, top_k: int) -> list[str]:
        """Keyword-based retrieval fallback."""
        query_words = set(re.findall(r"\w+", query.lower()))

        scored = []
        for passage in self._corpus:
            text = passage["text"].lower()
            passage_words = set(re.findall(r"\w+", text))
            overlap = len(query_words & passage_words)
            if overlap > 0:
                scored.append((overlap, passage["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def build_index(self, output_path: Path) -> None:
        """Build a FAISS index from the loaded corpus.

        Run this once to create the index file:
            python -m jarvis build-index --corpus data/physics_corpus.json --output data/physics.index
        """
        if not self._corpus:
            raise ValueError("No corpus loaded. Load corpus first.")

        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                f"Building FAISS index requires faiss-cpu and sentence-transformers: {e}"
            )

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [p["text"] for p in self._corpus]

        logger.info("Embedding %d passages...", len(texts))
        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        faiss.write_index(index, str(output_path))
        logger.info("FAISS index saved to %s (%d vectors, %d dimensions)", output_path, len(texts), dimension)
