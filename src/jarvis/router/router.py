"""Unified router — orchestrates domain classification, difficulty estimation,
and HEP subdomain detection into a single routing decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jarvis.config import JarvisConfig
from jarvis.router.difficulty_estimator import DifficultyEstimator, DifficultyResult
from jarvis.router.domain_classifier import ClassificationResult, DomainClassifier
from jarvis.router.hep_detector import HEPDetector

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """The complete routing decision for a query."""

    domain: str
    difficulty: str
    is_hep: bool
    base_model: str | None
    adapter: str | None
    specialist: str | None
    domain_confidence: float
    difficulty_confidence: float
    domain_method: str
    difficulty_method: str


class Router:
    """Routes incoming queries to the correct brain/specialist and difficulty level."""

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self._domain_classifier = DomainClassifier(config.router)
        self._difficulty_estimator = DifficultyEstimator(config.router)
        self._hep_detector = HEPDetector(config.router)

    def load(self) -> None:
        """Load BERT models if available, otherwise use bootstraps."""
        self._domain_classifier.load()
        self._difficulty_estimator.load()

    def route(
        self,
        query: str,
        system_prompt: str | None = None,
        force_domain: str | None = None,
    ) -> RoutingDecision:
        """Route a query to the correct brain and difficulty level.

        Args:
            query: The user's query text.
            system_prompt: Optional system prompt for context.
            force_domain: If set, skip domain classification and use this domain.
        """
        # Step 1: Domain classification (or forced)
        if force_domain and force_domain != "auto":
            domain_result = ClassificationResult(
                domain=force_domain, confidence=1.0, method="forced"
            )
        else:
            domain_result = self._domain_classifier.classify(query, system_prompt)

        domain = domain_result.domain

        # Step 2: HEP subdomain detection (only for physics/code)
        is_hep = False
        if domain in ("physics", "code") and self.config.router.hep_subdomain.enabled:
            is_hep = self._hep_detector.detect(query)

        # Step 3: Difficulty estimation
        difficulty_result = self._difficulty_estimator.estimate(query, domain)

        # Step 4: Resolve domain → brain/specialist via config mappings
        base_model, adapter, specialist = self._resolve_brain(domain, is_hep)

        decision = RoutingDecision(
            domain=domain,
            difficulty=difficulty_result.level,
            is_hep=is_hep,
            base_model=base_model,
            adapter=adapter,
            specialist=specialist,
            domain_confidence=domain_result.confidence,
            difficulty_confidence=difficulty_result.confidence,
            domain_method=domain_result.method,
            difficulty_method=difficulty_result.method,
        )

        logger.info(
            "Routed query: domain=%s (%.2f, %s), difficulty=%s (%.2f, %s), hep=%s, model=%s, adapter=%s",
            decision.domain,
            decision.domain_confidence,
            decision.domain_method,
            decision.difficulty,
            decision.difficulty_confidence,
            decision.difficulty_method,
            decision.is_hep,
            decision.base_model or decision.specialist,
            decision.adapter,
        )

        return decision

    def _resolve_brain(
        self, domain: str, is_hep: bool
    ) -> tuple[str | None, str | None, str | None]:
        """Map domain + HEP flag to base_model, adapter, specialist."""
        mapping = self.config.router.domain_to_brain.get(domain)
        if mapping is None:
            # Fall back to general
            mapping = self.config.router.domain_to_brain.get("general")
            if mapping is None:
                return None, None, None

        if mapping.specialist:
            return None, None, mapping.specialist

        base_model = mapping.base_model
        adapter = mapping.hep_adapter if is_hep and mapping.hep_adapter else mapping.adapter

        return base_model, adapter, None
