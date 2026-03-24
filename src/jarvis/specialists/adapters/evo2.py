"""Adapter for Evo 2 DNA language model."""

from __future__ import annotations

from typing import Any

from jarvis.specialists.adapters.base import SpecialistAdapter


class Evo2Adapter(SpecialistAdapter):
    """Translates chat messages containing DNA sequences to Evo 2 input format."""

    def parse_input(self, messages: list[dict]) -> Any:
        raise NotImplementedError("Phase 5: extract DNA sequence from chat messages")

    def format_output(self, model_output: Any) -> str:
        raise NotImplementedError("Phase 5: format Evo 2 output as chat response")
