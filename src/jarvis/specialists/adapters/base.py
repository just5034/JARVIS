"""Base class for specialist API adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SpecialistAdapter(ABC):
    """Translates between OpenAI chat format and specialist model native I/O."""

    @abstractmethod
    def parse_input(self, messages: list[dict]) -> Any:
        """Extract model-specific input from chat messages."""
        ...

    @abstractmethod
    def format_output(self, model_output: Any) -> str:
        """Convert model-specific output to a chat response string."""
        ...
