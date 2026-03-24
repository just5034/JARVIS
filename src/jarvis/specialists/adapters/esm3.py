"""Adapter for ESM3 protein language model."""

from __future__ import annotations

from typing import Any

from jarvis.specialists.adapters.base import SpecialistAdapter


class ESM3Adapter(SpecialistAdapter):
    """Translates chat messages containing protein sequences to ESM3 input format."""

    def parse_input(self, messages: list[dict]) -> Any:
        raise NotImplementedError("Phase 5: extract protein sequence from chat messages")

    def format_output(self, model_output: Any) -> str:
        raise NotImplementedError("Phase 5: format ESM3 output as chat response")
