"""Self-consistency voting — majority vote over extracted answers."""

from __future__ import annotations


class SelfConsistencyVoter:
    """Selects the most common final answer from N candidates."""

    def vote(self, candidates: list[str]) -> tuple[str, float]:
        raise NotImplementedError("Phase 3: self-consistency voting")
