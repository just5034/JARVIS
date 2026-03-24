"""Domain classification — routes queries to the correct brain/specialist."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.config import RouterConfig


@dataclass
class ClassificationResult:
    domain: str
    confidence: float
    is_hep: bool = False


class DomainClassifier:
    """Two-stage BERT classifier: domain + HEP subdomain detection."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._model = None

    def load(self) -> None:
        raise NotImplementedError("Phase 2: load fine-tuned BERT classifier")

    def classify(self, query: str, system_prompt: str | None = None) -> ClassificationResult:
        raise NotImplementedError("Phase 2: run domain classification")
