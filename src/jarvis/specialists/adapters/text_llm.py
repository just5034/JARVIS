"""Adapter for standard text LLM specialists (ChemLLM, BioMistral)."""

from __future__ import annotations

from typing import Any

from jarvis.specialists.adapters.base import SpecialistAdapter


class TextLLMAdapter(SpecialistAdapter):
    """Pass-through adapter for specialists that use standard chat format."""

    def parse_input(self, messages: list[dict]) -> Any:
        return messages

    def format_output(self, model_output: Any) -> str:
        return str(model_output)
