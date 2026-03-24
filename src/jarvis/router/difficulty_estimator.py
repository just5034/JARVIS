"""Difficulty estimation — determines inference strategy for a query.

Bootstrap implementation uses heuristics (query length, domain, complexity
signals). A trained BERT classifier can be loaded to replace the heuristic
approach (Phase 4).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from jarvis.config import RouterConfig

logger = logging.getLogger(__name__)

# Complexity signals that push difficulty upward
_HARD_SIGNALS = [
    r"\bprove\b",
    r"\bderive\b",
    r"\bshow\s+that\b",
    r"\bdemonstrate\b",
    r"\bfrom\s+first\s+principles",
    r"\brigorously",
    r"\boptimi[sz]e\b",
    r"\bminimize\b",
    r"\bmaximize\b",
    r"\bnumerically\b",
    r"\banalytically\b",
    r"\bmulti.?step",
    r"\bcomplex\b",
    r"\badvanced\b",
    r"\bchallenging\b",
    r"\bnon.?trivial",
    r"\bgeneral\s+case",
    r"\barbitrary\b",
    r"\bn\s*dimensions",
    r"\border\s+of\s+magnitude",
    r"\bloop\s+level",
    r"\bnlo\b",
    r"\bnnlo\b",
    r"\brendering\b",
    r"\brenormali[sz]",
]

_EASY_SIGNALS = [
    r"\bwhat\s+is\b",
    r"\bdefine\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\blist\b",
    r"\bname\b",
    r"\bwho\b",
    r"\bwhen\b",
    r"\bsimple\b",
    r"\bbasic\b",
    r"\bbriefly\b",
    r"\bquick\b",
    r"\bsummar",
    r"\bone.?line",
    r"\bhello\b",
    r"\bhi\b",
    r"\bthanks\b",
]

# Domains that tend toward higher difficulty
_HARD_DOMAINS = {"physics", "math"}
_EASY_DOMAINS = {"general"}


@dataclass
class DifficultyResult:
    level: str  # "easy", "medium", "hard"
    confidence: float
    method: str = "heuristic"  # "heuristic" or "bert"


class DifficultyEstimator:
    """Difficulty estimator with heuristic bootstrap and optional BERT model."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._model = None
        self._use_bert = False
        self._hard_patterns = [re.compile(p, re.IGNORECASE) for p in _HARD_SIGNALS]
        self._easy_patterns = [re.compile(p, re.IGNORECASE) for p in _EASY_SIGNALS]

    def load(self) -> None:
        """Load a trained BERT classifier if available."""
        checkpoint = self.config.difficulty_estimator.checkpoint_path
        if os.path.exists(checkpoint):
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                self._use_bert = True
                logger.info("Loaded BERT difficulty estimator from %s", checkpoint)
            except Exception as e:
                logger.warning("Failed to load BERT estimator, using heuristics: %s", e)
        else:
            logger.info("No BERT checkpoint at %s, using heuristic estimator", checkpoint)

    def estimate(self, query: str, domain: str) -> DifficultyResult:
        """Estimate the difficulty of a query."""
        if self._use_bert and self._model is not None:
            return self._estimate_bert(query, domain)
        return self._estimate_heuristic(query, domain)

    def _estimate_heuristic(self, query: str, domain: str) -> DifficultyResult:
        """Heuristic-based difficulty estimation."""
        hard_hits = sum(1 for p in self._hard_patterns if p.search(query))
        easy_hits = sum(1 for p in self._easy_patterns if p.search(query))

        # Length-based signal
        word_count = len(query.split())
        length_score = 0
        if word_count > 200:
            length_score = 2
        elif word_count > 80:
            length_score = 1
        elif word_count < 15:
            length_score = -1

        # Domain bias
        domain_score = 0
        if domain in _HARD_DOMAINS:
            domain_score = 1
        elif domain in _EASY_DOMAINS:
            domain_score = -1

        # Aggregate
        score = hard_hits - easy_hits + length_score + domain_score

        if score >= 3:
            level = "hard"
            confidence = min(0.5 + score * 0.1, 0.9)
        elif score <= -1:
            level = "easy"
            confidence = min(0.5 + abs(score) * 0.1, 0.9)
        else:
            level = "medium"
            confidence = 0.5

        return DifficultyResult(level=level, confidence=confidence, method="heuristic")

    def _estimate_bert(self, query: str, domain: str) -> DifficultyResult:
        """BERT-based difficulty estimation (Phase 4)."""
        import torch

        text = f"[{domain}] {query}"
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, idx = probs.max(dim=-1)

        levels = self.config.difficulty_estimator.levels
        level = levels[idx.item()] if idx.item() < len(levels) else "medium"

        return DifficultyResult(
            level=level,
            confidence=confidence.item(),
            method="bert",
        )
