"""Adapter for ESM3 protein language model.

Extracts amino acid sequences from chat messages and formats
structure/function predictions as natural language responses.
"""

from __future__ import annotations

import re
from typing import Any

from jarvis.specialists.adapters.base import SpecialistAdapter

# Standard amino acid single-letter codes
_AA_PATTERN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{10,}", re.IGNORECASE)

# Common sequence prefixes in user messages
_SEQUENCE_PATTERNS = [
    re.compile(r"sequence[:\s]+([A-Z]{10,})", re.IGNORECASE),
    re.compile(r"protein[:\s]+([A-Z]{10,})", re.IGNORECASE),
    re.compile(r">.*?\n([A-Z\s]+)", re.IGNORECASE),  # FASTA format
]


class ESM3Adapter(SpecialistAdapter):
    """Translates chat messages containing protein sequences to ESM3 input format."""

    def parse_input(self, messages: list[dict]) -> dict:
        """Extract protein sequence and task from chat messages.

        Returns dict with 'sequence' and 'task' keys.
        """
        # Combine all user messages
        user_text = " ".join(
            m["content"] for m in messages if m.get("role") == "user"
        )

        # Try specific patterns first
        sequence = None
        for pattern in _SEQUENCE_PATTERNS:
            match = pattern.search(user_text)
            if match:
                sequence = match.group(1).replace(" ", "").replace("\n", "").upper()
                break

        # Fall back to finding any long amino acid sequence
        if sequence is None:
            match = _AA_PATTERN.search(user_text)
            if match:
                sequence = match.group(0).upper()

        # Determine the task
        task = "general"
        text_lower = user_text.lower()
        if any(w in text_lower for w in ("structure", "fold", "3d", "conformation")):
            task = "structure_prediction"
        elif any(w in text_lower for w in ("function", "annotation", "role")):
            task = "function_prediction"
        elif any(w in text_lower for w in ("embed", "representation", "vector")):
            task = "embedding"

        return {
            "sequence": sequence,
            "task": task,
            "raw_query": user_text,
        }

    def format_output(self, model_output: Any) -> str:
        """Format ESM3 model output as a natural language response."""
        if model_output is None:
            return "Unable to process the protein sequence. Please ensure a valid amino acid sequence was provided."

        if isinstance(model_output, dict):
            parts = []

            if "sequence" in model_output:
                seq = model_output["sequence"]
                parts.append(f"**Input Sequence** ({len(seq)} residues): `{seq[:50]}{'...' if len(seq) > 50 else ''}`")

            if "structure" in model_output:
                parts.append(f"\n**Structure Prediction:**\n{model_output['structure']}")

            if "function" in model_output:
                parts.append(f"\n**Function Prediction:**\n{model_output['function']}")

            if "confidence" in model_output:
                parts.append(f"\n**Confidence:** {model_output['confidence']:.2f}")

            if "embeddings" in model_output:
                emb = model_output["embeddings"]
                parts.append(f"\n**Embedding:** {len(emb)}-dimensional vector generated")

            return "\n".join(parts) if parts else str(model_output)

        return str(model_output)
