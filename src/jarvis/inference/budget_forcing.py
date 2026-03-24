"""Budget forcing — the 'Wait' trick for extended self-examination."""

from __future__ import annotations


class BudgetForcer:
    """Appends 'Wait' tokens to force continued reasoning on hard queries."""

    def __init__(self, max_waits: int = 3) -> None:
        self.max_waits = max_waits

    def should_force(self, output: str, thinking_tokens: int, budget: int) -> bool:
        raise NotImplementedError("Phase 3: detect premature conclusion")

    def apply(self, output: str) -> str:
        raise NotImplementedError("Phase 3: append 'Wait' and continue generation")
