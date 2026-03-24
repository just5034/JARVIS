"""Inference engine — difficulty-aware strategy selection and execution."""

from __future__ import annotations

from jarvis.config import InferenceConfig


class InferenceEngine:
    """Selects and runs the inference strategy based on difficulty level."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    async def generate(
        self,
        model: object,
        messages: list[dict],
        difficulty: str,
        domain: str,
    ) -> dict:
        raise NotImplementedError("Phase 3: strategy selection and execution")
