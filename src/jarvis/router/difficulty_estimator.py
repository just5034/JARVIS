"""Difficulty estimation — determines inference strategy for a query."""

from __future__ import annotations

from jarvis.config import RouterConfig


class DifficultyEstimator:
    """BERT classifier that predicts easy/medium/hard difficulty."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._model = None

    def load(self) -> None:
        raise NotImplementedError("Phase 2: load fine-tuned difficulty classifier")

    def estimate(self, query: str, domain: str) -> str:
        raise NotImplementedError("Phase 2: run difficulty estimation")
