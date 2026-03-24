"""Adapter for Evo 2 DNA language model.

Extracts DNA sequences from chat messages and formats variant effect
predictions and generation results as natural language responses.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.specialists.adapters.base import SpecialistAdapter

# DNA sequence pattern (at least 20 nucleotides)
_DNA_PATTERN = re.compile(r"[ACGTacgt]{20,}")

# Common sequence prefixes
_SEQUENCE_PATTERNS = [
    re.compile(r"sequence[:\s]+([ACGTacgt]{20,})", re.IGNORECASE),
    re.compile(r"DNA[:\s]+([ACGTacgt]{20,})", re.IGNORECASE),
    re.compile(r">.*?\n([ACGTacgt\s]+)", re.IGNORECASE),  # FASTA format
]

# Mutation pattern: e.g., "G>A at position 12345" or "A123G"
_MUTATION_PATTERN = re.compile(
    r"(?:([ACGT])>([ACGT])\s*(?:at\s*)?(?:position\s*)?(\d+))|"
    r"(?:([ACGT])(\d+)([ACGT]))",
    re.IGNORECASE,
)


class Evo2Adapter(SpecialistAdapter):
    """Translates chat messages containing DNA sequences to Evo 2 input format."""

    def parse_input(self, messages: list[dict]) -> dict:
        """Extract DNA sequence, mutations, and task from chat messages."""
        user_text = " ".join(
            m["content"] for m in messages if m.get("role") == "user"
        )

        # Extract sequence
        sequence = None
        for pattern in _SEQUENCE_PATTERNS:
            match = pattern.search(user_text)
            if match:
                sequence = match.group(1).replace(" ", "").replace("\n", "").upper()
                break

        if sequence is None:
            match = _DNA_PATTERN.search(user_text)
            if match:
                sequence = match.group(0).upper()

        # Extract mutations
        mutations = []
        for match in _MUTATION_PATTERN.finditer(user_text):
            if match.group(1):  # G>A at position 12345
                mutations.append({
                    "ref": match.group(1).upper(),
                    "alt": match.group(2).upper(),
                    "position": int(match.group(3)),
                })
            elif match.group(4):  # A123G format
                mutations.append({
                    "ref": match.group(4).upper(),
                    "alt": match.group(6).upper(),
                    "position": int(match.group(5)),
                })

        # Determine the task
        task = "general"
        text_lower = user_text.lower()
        if mutations or any(w in text_lower for w in ("mutation", "variant", "effect", "pathogenic")):
            task = "variant_effect"
        elif any(w in text_lower for w in ("generate", "design", "create", "synthesize")):
            task = "sequence_generation"
        elif any(w in text_lower for w in ("embed", "representation")):
            task = "embedding"

        return {
            "sequence": sequence,
            "mutations": mutations,
            "task": task,
            "raw_query": user_text,
        }

    def format_output(self, model_output: Any) -> str:
        """Format Evo 2 model output as a natural language response."""
        if model_output is None:
            return "Unable to process the DNA sequence. Please ensure a valid nucleotide sequence was provided."

        if isinstance(model_output, dict):
            parts = []

            if "sequence" in model_output:
                seq = model_output["sequence"]
                parts.append(f"**Input Sequence** ({len(seq)} bp): `{seq[:50]}{'...' if len(seq) > 50 else ''}`")

            if "variant_effects" in model_output:
                parts.append("\n**Variant Effect Predictions:**")
                for ve in model_output["variant_effects"]:
                    ref = ve.get("ref", "?")
                    alt = ve.get("alt", "?")
                    pos = ve.get("position", "?")
                    score = ve.get("score", 0)
                    label = ve.get("label", "uncertain")
                    parts.append(f"- {ref}{pos}{alt}: {label} (score: {score:.3f})")

            if "generated_sequence" in model_output:
                gen = model_output["generated_sequence"]
                parts.append(f"\n**Generated Sequence** ({len(gen)} bp): `{gen[:80]}{'...' if len(gen) > 80 else ''}`")

            if "log_likelihood" in model_output:
                parts.append(f"\n**Log-likelihood:** {model_output['log_likelihood']:.4f}")

            return "\n".join(parts) if parts else str(model_output)

        return str(model_output)
