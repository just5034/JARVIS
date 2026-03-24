"""Best-of-N candidate generation."""

from __future__ import annotations


class CandidateSampler:
    """Generates N candidate responses in parallel."""

    async def sample(self, model: object, messages: list[dict], n: int) -> list[str]:
        raise NotImplementedError("Phase 3: parallel candidate generation")
