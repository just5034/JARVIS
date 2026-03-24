"""On-demand specialist model loading with LRU eviction."""

from __future__ import annotations

from collections import OrderedDict

from jarvis.brains.memory_tracker import MemoryTracker
from jarvis.config import SpecialistConfig


class SpecialistLoader:
    """Loads specialist models from SSD on demand, evicting LRU when needed."""

    def __init__(self, memory: MemoryTracker) -> None:
        self.memory = memory
        self._loaded: OrderedDict[str, object] = OrderedDict()

    async def load(self, name: str, config: SpecialistConfig) -> object:
        raise NotImplementedError("Phase 5: load specialist model from SSD")

    async def unload_lru(self) -> str | None:
        raise NotImplementedError("Phase 5: evict least-recently-used specialist")

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded
