"""Best-of-N candidate generation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.brains.model_loader import GenerationResult, LoadedModelHandle


class CandidateSampler:
    """Generates N candidate responses via vLLM's parallel sampling."""

    async def sample(
        self,
        model: "LoadedModelHandle",
        messages: list[dict],
        n: int,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> list["GenerationResult"]:
        """Generate N candidate responses.

        Uses vLLM's native n parameter for efficient batch generation.
        Wrapped in asyncio.to_thread to avoid blocking the event loop.
        """
        from jarvis.brains.model_loader import GenerationRequest

        request = GenerationRequest(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
        )

        results = await asyncio.to_thread(model.generate, request)
        return results
