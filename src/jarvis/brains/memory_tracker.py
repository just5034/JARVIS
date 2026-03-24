"""Memory tracking — RAM budget ledger for model loading decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.config import MemoryBudgetConfig


@dataclass
class LoadedModel:
    key: str
    size_gb: float
    model_type: str  # "base", "adapter", "specialist", "infrastructure"


class MemoryTracker:
    """Tracks memory usage and enforces the RAM budget."""

    def __init__(self, budget: MemoryBudgetConfig) -> None:
        self.budget = budget
        self._loaded: dict[str, LoadedModel] = {}

    @property
    def used_gb(self) -> float:
        return sum(m.size_gb for m in self._loaded.values())

    @property
    def available_gb(self) -> float:
        return self.budget.available_gb - self.used_gb

    def can_load(self, size_gb: float) -> bool:
        return size_gb <= self.available_gb

    def register(self, key: str, size_gb: float, model_type: str) -> None:
        if not self.can_load(size_gb):
            raise MemoryError(
                f"Cannot load '{key}' ({size_gb:.1f} GB): "
                f"only {self.available_gb:.1f} GB available"
            )
        self._loaded[key] = LoadedModel(key=key, size_gb=size_gb, model_type=model_type)

    def unregister(self, key: str) -> None:
        self._loaded.pop(key, None)

    def summary(self) -> list[dict]:
        return [
            {"name": m.key, "size_gb": m.size_gb, "type": m.model_type}
            for m in self._loaded.values()
        ]
