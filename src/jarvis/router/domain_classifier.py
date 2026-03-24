"""Domain classification — routes queries to the correct brain/specialist.

Bootstrap implementation uses keyword matching. A trained BERT classifier
can be loaded to replace the keyword approach (Phase 4).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from jarvis.config import RouterConfig

logger = logging.getLogger(__name__)

# Domain keyword patterns — order matters (more specific domains first)
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "protein": [
        r"\bamino acid",
        r"\bprotein\b",
        r"\bpeptide",
        r"\bsequence.*fold",
        r"\bstructure prediction",
        r"\balpha.?fold",
        r"\besm\d",
        r"\bprotein.*structure",
    ],
    "genomics": [
        r"\bdna\b",
        r"\brna\b",
        r"\bgenom",
        r"\bnucleotide",
        r"\bmutation.*effect",
        r"\bvariant.*effect",
        r"\bgene\b",
        r"\bchromosome",
        r"\bcrispr",
        r"\bsequencing",
        r"\bbrca",
    ],
    "chemistry": [
        r"\breaction\b",
        r"\bmolecul",
        r"\bcompound",
        r"\bsynthesi[sz]",
        r"\bcatalys",
        r"\bsolvent",
        r"\bchemical",
        r"\borganic\b",
        r"\binorganic\b",
        r"\bpolymer",
        r"\breagent",
        r"\bstoichiometr",
        r"\bmolar\b",
        r"\btitrat",
    ],
    "biology": [
        r"\bcell\b",
        r"\bbiomedic",
        r"\bclinical",
        r"\bdisease",
        r"\bdrug\b",
        r"\bpathway",
        r"\btissue",
        r"\borgan\b",
        r"\bimmun",
        r"\bmetaboli",
        r"\bbiolog",
        r"\bmicrob",
    ],
    "physics": [
        r"\bphysic",
        r"\bquantum\b",
        r"\brelativi",
        r"\belectromagnet",
        r"\bthermodynamic",
        r"\bhamiltonian",
        r"\blagrangian",
        r"\bschrodinger",
        r"\bnewton",
        r"\bplanck",
        r"\bwave\s*function",
        r"\bfield\s*theory",
        r"\bcross.?section",
        r"\bdecay\s*width",
        r"\bfeynman",
        r"\bstandard\s*model",
        r"\bgev\b",
        r"\bmev\b",
        r"\btev\b",
        r"\bluminosity",
        r"\bparticle\b",
        r"\bboson",
        r"\bfermion",
        r"\blepton",
        r"\bhadron",
        r"\bneutrino",
        r"\bscattering",
        r"\bdetector",
        r"\bcalorimeter",
        r"\boptic(?:s|al)",
        r"\bkinematic",
        r"\bmomentum",
        r"\benergy.*conservation",
        r"\bentropy\b",
    ],
    "math": [
        r"\bprove\b",
        r"\bproof\b",
        r"\btheorem\b",
        r"\blemma\b",
        r"\bcorollar",
        r"\bintegr(?:al|ate)",
        r"\bderivativ",
        r"\bdifferential\s*equation",
        r"\bmatrix\b",
        r"\beigenvalue",
        r"\btopolog",
        r"\balgebra\b",
        r"\bcombinatoric",
        r"\bprobabilit",
        r"\bstatistic",
        r"\bnumber\s*theory",
        r"\bprime\b.*\bnumber",
        r"\bconvergence",
        r"\bseries\b",
        r"\blimit\b",
        r"\bcalcul(?:us|ate)",
        r"\bsolve\b.*\bequation",
        r"\bfind\b.*\bvalue",
        r"\bcompute\b",
    ],
    "code": [
        r"\bcode\b",
        r"\bprogram",
        r"\bfunction\b",
        r"\bclass\b",
        r"\bimplement",
        r"\bdebug",
        r"\brefactor",
        r"\bpython\b",
        r"\bjava\b",
        r"\bc\+\+",
        r"\brust\b",
        r"\bjavascript\b",
        r"\btypescript\b",
        r"\bapi\b",
        r"\balgorithm\b",
        r"\bdata\s*structure",
        r"\bsort\b",
        r"\bsearch\b",
        r"\btree\b",
        r"\bgraph\b.*\btravers",
        r"\brecursion\b",
        r"\bloop\b",
        r"\barray\b",
        r"\bstring\b.*\b(?:manipulat|pars)",
        r"\bsql\b",
        r"\bdatabase\b",
        r"\bgit\b",
        r"\bdocker\b",
        r"\bwrite\b.*\b(?:script|code|program|function)",
    ],
}


@dataclass
class ClassificationResult:
    domain: str
    confidence: float
    is_hep: bool = False
    method: str = "keyword"  # "keyword" or "bert"


class DomainClassifier:
    """Domain classifier with keyword bootstrap and optional BERT model.

    The keyword approach works as a reasonable bootstrap until a trained
    BERT classifier is available from Phase 4.
    """

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._model = None
        self._use_bert = False
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}

        # Pre-compile regex patterns
        for domain, patterns in _DOMAIN_KEYWORDS.items():
            if domain in config.domain_classifier.domains:
                self._compiled_patterns[domain] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

    def load(self) -> None:
        """Load a trained BERT classifier if available."""
        checkpoint = self.config.domain_classifier.checkpoint_path
        if os.path.exists(checkpoint):
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
                self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                self._use_bert = True
                logger.info("Loaded BERT domain classifier from %s", checkpoint)
            except Exception as e:
                logger.warning("Failed to load BERT classifier, using keywords: %s", e)
        else:
            logger.info("No BERT checkpoint at %s, using keyword classifier", checkpoint)

    def classify(self, query: str, system_prompt: str | None = None) -> ClassificationResult:
        """Classify a query into a domain."""
        if self._use_bert and self._model is not None:
            return self._classify_bert(query, system_prompt)
        return self._classify_keywords(query, system_prompt)

    def _classify_keywords(
        self, query: str, system_prompt: str | None = None
    ) -> ClassificationResult:
        """Keyword-based domain classification."""
        text = query
        if system_prompt:
            text = f"{system_prompt} {query}"

        scores: dict[str, int] = {}
        for domain, patterns in self._compiled_patterns.items():
            score = sum(1 for p in patterns if p.search(text))
            if score > 0:
                scores[domain] = score

        if not scores:
            return ClassificationResult(
                domain=self.config.domain_classifier.fallback_domain,
                confidence=0.0,
                method="keyword",
            )

        best_domain = max(scores, key=scores.__getitem__)
        best_score = scores[best_domain]
        total_matches = sum(scores.values())

        confidence = min(best_score / max(total_matches, 1), 1.0)

        if confidence < self.config.domain_classifier.confidence_threshold:
            # Check if there's a close second — if so, fall back
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] <= 1:
                return ClassificationResult(
                    domain=self.config.domain_classifier.fallback_domain,
                    confidence=confidence,
                    method="keyword",
                )

        return ClassificationResult(
            domain=best_domain,
            confidence=confidence,
            method="keyword",
        )

    def _classify_bert(
        self, query: str, system_prompt: str | None = None
    ) -> ClassificationResult:
        """BERT-based domain classification (Phase 4)."""
        import torch

        text = query
        if system_prompt:
            text = f"{system_prompt} [SEP] {query}"

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, idx = probs.max(dim=-1)

        domains = self.config.domain_classifier.domains
        domain = domains[idx.item()] if idx.item() < len(domains) else "general"
        conf = confidence.item()

        if conf < self.config.domain_classifier.confidence_threshold:
            domain = self.config.domain_classifier.fallback_domain

        return ClassificationResult(
            domain=domain,
            confidence=conf,
            method="bert",
        )
