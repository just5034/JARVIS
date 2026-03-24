"""KV cache management — FP8/2-bit quantization, SSD offload."""

from __future__ import annotations

from jarvis.config import ContextManagementConfig


class ContextManager:
    """Manages KV cache configuration per inference strategy."""

    def __init__(self, config: ContextManagementConfig) -> None:
        self.config = config

    def get_kv_config(self, difficulty: str, num_candidates: int) -> dict:
        raise NotImplementedError("Phase 3: compute KV cache settings for strategy")
