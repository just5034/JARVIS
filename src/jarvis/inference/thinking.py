"""Thinking format handling for different model families.

Qwen3.5-27B outputs reasoning as visible "Thinking Process:" text blocks.
R1-Distill models use <think>...</think> XML tags.
This module normalizes both formats for downstream processing.
"""

from __future__ import annotations

import re

# Qwen3.5 pattern: "Thinking Process:" followed by content, ending at double newline
# before the actual answer. The thinking block is typically the first section.
_QWEN35_THINKING_RE = re.compile(
    r"^Thinking Process:\s*\n(.*?)(?:\n\n|\Z)",
    re.DOTALL,
)

# R1-Distill pattern: <think>...</think> XML tags
_R1_THINKING_RE = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)


def split_thinking(text: str) -> tuple[str, str]:
    """Split model output into (thinking, answer).

    Returns (thinking_text, answer_text). If no thinking block is detected,
    returns ("", original_text).
    """
    # Try R1-Distill format first (more specific)
    m = _R1_THINKING_RE.search(text)
    if m:
        thinking = m.group(1).strip()
        answer = text[:m.start()] + text[m.end():]
        return thinking, answer.strip()

    # Try Qwen3.5 format
    m = _QWEN35_THINKING_RE.match(text)
    if m:
        thinking = m.group(1).strip()
        answer = text[m.end():].strip()
        return thinking, answer

    return "", text


def strip_thinking(text: str) -> str:
    """Remove thinking blocks, returning only the answer portion."""
    _, answer = split_thinking(text)
    return answer


def has_thinking(text: str) -> bool:
    """Check if the text contains a thinking block."""
    return bool(
        _R1_THINKING_RE.search(text) or _QWEN35_THINKING_RE.match(text)
    )


# Conclusion patterns that indicate the model has finished reasoning.
# Used by budget forcing to detect premature conclusions.
CONCLUSION_PATTERNS = [
    # R1-Distill: closing think tag
    re.compile(r"</think>", re.IGNORECASE),
    # Qwen3.5: transition from thinking to answer (double newline after thinking block)
    re.compile(r"Thinking Process:.*?\n\n\S", re.DOTALL),
    # Universal patterns
    re.compile(r"\\boxed\{", re.IGNORECASE),
    re.compile(r"\*\*Final Answer\*\*", re.IGNORECASE),
    re.compile(r"(?:^|\n)(?:Therefore|Thus|Hence|In conclusion),?\s", re.IGNORECASE),
    re.compile(r"(?:^|\n)The answer is\s", re.IGNORECASE),
]
