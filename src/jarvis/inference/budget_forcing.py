"""Budget forcing — the 'Wait' trick for extended self-examination.

When a model produces a conclusion before exhausting its thinking budget,
we strip the conclusion and append a continuation prompt to force deeper
reasoning. Up to max_waits iterations.
"""

from __future__ import annotations

import re

# Patterns that indicate premature conclusion
_CONCLUSION_PATTERNS = [
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"\\boxed\{", re.IGNORECASE),
    re.compile(r"\*\*Final Answer\*\*", re.IGNORECASE),
    re.compile(r"(?:^|\n)(?:Therefore|Thus|Hence|In conclusion),?\s", re.IGNORECASE),
    re.compile(r"(?:^|\n)The answer is\s", re.IGNORECASE),
]

_CONTINUATION_PROMPT = "\nWait, let me reconsider this step.\n"


class BudgetForcer:
    """Appends continuation prompts to force extended reasoning on hard queries."""

    def __init__(self, max_waits: int = 3) -> None:
        self.max_waits = max_waits
        self._force_count = 0

    @property
    def force_count(self) -> int:
        return self._force_count

    def reset(self) -> None:
        """Reset the force counter for a new query."""
        self._force_count = 0

    def should_force(self, output: str, thinking_tokens: int, budget: int) -> bool:
        """Check if we should force continued reasoning.

        Forces continuation when:
        1. We haven't exceeded max_waits
        2. Token count is below 50% of the thinking budget
        3. Output contains a premature conclusion marker
        """
        if self._force_count >= self.max_waits:
            return False

        if budget <= 0:
            return False

        if thinking_tokens >= budget * 0.5:
            return False

        for pattern in _CONCLUSION_PATTERNS:
            if pattern.search(output):
                return True

        return False

    def apply(self, output: str) -> str:
        """Strip premature conclusion and append continuation prompt.

        Returns the modified output to be used as context for re-generation.
        """
        modified = output.rstrip()

        # Strip </think> tag if present
        modified = re.sub(r"</think>\s*$", "", modified)

        modified += _CONTINUATION_PROMPT
        self._force_count += 1

        return modified
