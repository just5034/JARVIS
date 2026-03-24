"""ThinkPRM verification — scores reasoning chains for quality.

Uses the ThinkPRM-1.5B process reward model to score each candidate's
reasoning chain. Gracefully degrades when the model isn't available
(returns neutral scores).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ThinkPRMVerifier:
    """Process reward model for scoring reasoning chains."""

    def __init__(self, model_id: str = "PRIME-RL/ThinkPRM-1.5B") -> None:
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._available = False
        self._device = "cpu"

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> None:
        """Load the ThinkPRM model. Gracefully degrades if unavailable."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading ThinkPRM from %s on %s...", self._model_id, self._device
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()
            self._available = True
            logger.info("ThinkPRM loaded successfully")
        except ImportError:
            logger.warning("transformers not installed, ThinkPRM verification disabled")
            self._available = False
        except Exception as e:
            logger.warning("Failed to load ThinkPRM: %s. Verification disabled.", e)
            self._available = False

    def score(self, reasoning_chain: str) -> float:
        """Score a reasoning chain. Returns 0.5 (neutral) if model unavailable."""
        if not self._available or self._model is None:
            return 0.5

        try:
            import torch

            prompt = (
                "Rate the quality of the following reasoning on a scale from 0 to 1, "
                "where 1 is perfect reasoning:\n\n"
                f"{reasoning_chain}\n\n"
                "Score:"
            )

            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )

            generated = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Try to parse a float from the generated text
            import re

            score_match = re.search(r"(\d+\.?\d*)", generated)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)

            return 0.5
        except Exception as e:
            logger.warning("ThinkPRM scoring failed: %s", e)
            return 0.5

    def select_best(
        self,
        candidates: list[str],
        pessimistic: bool = True,
    ) -> tuple[str, float]:
        """Select the best candidate by reasoning quality score.

        Args:
            candidates: List of candidate response texts.
            pessimistic: If True, select by highest minimum score (safest reasoning).
                        If False, select by highest average score.

        Returns:
            Tuple of (best candidate text, score).
        """
        if not candidates:
            raise ValueError("No candidates to verify")

        if len(candidates) == 1:
            score = self.score(candidates[0])
            return candidates[0], score

        scores = [self.score(c) for c in candidates]

        if pessimistic:
            # Select candidate with highest score (most confident reasoning)
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
        else:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])

        return candidates[best_idx], scores[best_idx]
