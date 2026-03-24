"""RAG augmentation — prepends retrieved passages to the prompt."""

from __future__ import annotations


class PromptAugmenter:
    """Prepends retrieved knowledge passages to the user query."""

    def augment(self, messages: list[dict], passages: list[str]) -> list[dict]:
        raise NotImplementedError("Phase 6: prepend passages to prompt")
