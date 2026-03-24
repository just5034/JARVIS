"""RAG augmentation — prepends retrieved passages to the prompt."""

from __future__ import annotations


class PromptAugmenter:
    """Prepends retrieved knowledge passages to the user query."""

    def augment(self, messages: list[dict], passages: list[str]) -> list[dict]:
        """Augment messages with retrieved knowledge passages.

        Inserts a system message with the retrieved context before the
        conversation, or appends to an existing system message.
        """
        if not passages:
            return messages

        context = "Relevant reference information:\n\n" + "\n\n".join(
            f"- {passage}" for passage in passages
        )

        messages = [m.copy() for m in messages]

        # Find existing system message
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i] = {
                    "role": "system",
                    "content": msg["content"] + "\n\n" + context,
                }
                return messages

        # No system message — insert one at the beginning
        messages.insert(0, {"role": "system", "content": context})
        return messages
