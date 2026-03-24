"""ThinkPRM verification — scores reasoning chains for quality."""

from __future__ import annotations


class ThinkPRMVerifier:
    """Uses a process reward model to score candidate solutions."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        raise NotImplementedError("Phase 3: load ThinkPRM-1.5B")

    def score(self, reasoning_chain: str) -> float:
        raise NotImplementedError("Phase 3: score reasoning chain")

    def select_best(self, candidates: list[str], pessimistic: bool = True) -> str:
        raise NotImplementedError("Phase 3: pessimistic/optimistic selection")
