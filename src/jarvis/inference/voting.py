"""Self-consistency voting and domain-aware answer extraction."""

from __future__ import annotations

import re
from collections import Counter

from jarvis.inference.thinking import strip_thinking


class AnswerExtractor:
    """Extracts canonical answers from freeform model responses for voting."""

    def extract(self, text: str, domain: str) -> str:
        """Extract the canonical answer from a response, domain-aware.

        Strips thinking blocks (R1-Distill <think> tags or Qwen3.5
        "Thinking Process:" text) before extraction so we only
        match patterns in the actual answer portion.
        """
        text = strip_thinking(text)
        if domain == "math":
            return self._extract_math(text)
        elif domain == "code":
            return self._extract_code(text)
        else:
            return self._extract_general(text)

    def _extract_math(self, text: str) -> str:
        """Extract boxed answer or last numeric expression."""
        # Look for \boxed{...} (LaTeX convention for final answers)
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return boxed[-1].strip()

        # Look for "The answer is ..." pattern
        answer_match = re.search(
            r"(?:the\s+answer\s+is|therefore|thus|=)\s*([^\n,]+)",
            text,
            re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Fallback: last line with a number
        lines = text.strip().split("\n")
        for line in reversed(lines):
            if re.search(r"\d", line):
                return line.strip()

        return self._extract_general(text)

    def _extract_code(self, text: str) -> str:
        """Extract the last fenced code block."""
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Look for indented code blocks (4 spaces)
        indented = re.findall(r"(?:^    .+\n?)+", text, re.MULTILINE)
        if indented:
            return indented[-1].strip()

        return self._extract_general(text)

    def _extract_general(self, text: str) -> str:
        """Normalize text for general comparison."""
        # Strip whitespace, collapse spaces, lowercase
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        # Take the last paragraph as the "answer"
        paragraphs = normalized.split("\n\n")
        return paragraphs[-1].strip() if paragraphs else normalized


class SelfConsistencyVoter:
    """Selects the best response via majority vote over extracted answers."""

    def __init__(self) -> None:
        self._extractor = AnswerExtractor()

    def vote(self, candidates: list[str], domain: str) -> tuple[str, float]:
        """Vote over candidate responses.

        Returns:
            Tuple of (full text of winning candidate, confidence score).
        """
        if not candidates:
            raise ValueError("No candidates to vote on")

        if len(candidates) == 1:
            return candidates[0], 1.0

        # Extract canonical answers
        extracted = [self._extractor.extract(c, domain) for c in candidates]

        # Majority vote
        counts = Counter(extracted)
        winner_answer, winner_count = counts.most_common(1)[0]
        confidence = winner_count / len(candidates)

        # Return the full text of the first candidate that matches the winning answer
        for candidate, answer in zip(candidates, extracted):
            if answer == winner_answer:
                return candidate, confidence

        # Shouldn't reach here, but fallback
        return candidates[0], confidence
