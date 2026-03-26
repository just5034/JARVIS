"""Tests for the JARVIS router (domain classifier, difficulty estimator, unified router)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, load_config
from jarvis.router.domain_classifier import DomainClassifier
from jarvis.router.difficulty_estimator import DifficultyEstimator
from jarvis.router.hep_detector import HEPDetector
from jarvis.router.router import Router


@pytest.fixture
def config() -> JarvisConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    return load_config(config_dir)


# --- Domain Classifier ---


class TestDomainClassifier:
    def test_physics_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Calculate the Higgs boson decay width to b-bbar")
        assert result.domain == "physics"
        assert result.confidence > 0

    def test_math_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Prove that the sum of two even numbers is even")
        assert result.domain == "math"

    def test_code_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Write a Python function to sort a list using quicksort")
        assert result.domain == "code"

    def test_chemistry_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("What is the reaction mechanism for Fischer esterification?")
        assert result.domain == "chemistry"

    def test_biology_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Describe the metabolic pathway of glycolysis in human cells")
        assert result.domain == "biology"

    def test_protein_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Predict the structure of this protein sequence MVLSPADKTNVKAAW")
        assert result.domain == "protein"

    def test_genomics_query(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("What is the effect of BRCA1 mutation G>A at position 12345 on DNA?")
        assert result.domain == "genomics"

    def test_general_fallback(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Hello, how are you today?")
        assert result.domain == "general"

    def test_system_prompt_context(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify(
            "What is the cross-section?",
            system_prompt="You are a particle physics expert",
        )
        assert result.domain == "physics"

    def test_method_is_keyword(self, config: JarvisConfig) -> None:
        clf = DomainClassifier(config.router)
        result = clf.classify("Solve this integral")
        assert result.method == "keyword"


# --- Difficulty Estimator ---


class TestDifficultyEstimator:
    def test_easy_query(self, config: JarvisConfig) -> None:
        est = DifficultyEstimator(config.router)
        result = est.estimate("What is the speed of light?", "physics")
        assert result.level == "easy"

    def test_hard_query(self, config: JarvisConfig) -> None:
        est = DifficultyEstimator(config.router)
        result = est.estimate(
            "Derive from first principles the renormalization group equation "
            "for the running coupling constant in QCD at NLO, showing all "
            "intermediate steps rigorously and proving convergence",
            "physics",
        )
        assert result.level == "hard"

    def test_medium_default(self, config: JarvisConfig) -> None:
        est = DifficultyEstimator(config.router)
        result = est.estimate(
            "Calculate the eigenvalues of the following 3x3 matrix",
            "math",
        )
        assert result.level in ("medium", "hard")  # math domain bias pushes up

    def test_method_is_heuristic(self, config: JarvisConfig) -> None:
        est = DifficultyEstimator(config.router)
        result = est.estimate("Hello", "general")
        assert result.method == "heuristic"


# --- HEP Detector ---


class TestHEPDetector:
    def test_hep_detected(self, config: JarvisConfig) -> None:
        det = HEPDetector(config.router)
        assert det.detect("Calculate the Higgs boson mass") is True
        assert det.detect("Simulate the LHC detector response") is True
        assert det.detect("Run Geant4 simulation for calorimeter") is True

    def test_hep_not_detected(self, config: JarvisConfig) -> None:
        det = HEPDetector(config.router)
        assert det.detect("Solve the Schrodinger equation") is False
        assert det.detect("What is thermodynamics?") is False


# --- Unified Router ---


class TestRouter:
    def test_physics_routing(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("Calculate the decay width of the Higgs boson")
        assert decision.domain == "physics"
        assert decision.base_model == "r1_distill_qwen_32b"
        assert decision.specialist is None

    def test_code_routing(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("Write a Python function to implement binary search")
        assert decision.domain == "code"
        assert decision.base_model == "qwen25_coder_32b"

    def test_math_routing(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("Prove that there are infinitely many prime numbers")
        assert decision.domain == "math"
        assert decision.base_model == "r1_distill_qwen_32b"
        assert decision.adapter == "math_adapter"

    def test_chemistry_routes_to_specialist(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("What is the reaction mechanism for SN2?")
        assert decision.domain == "chemistry"
        assert decision.specialist == "chemistry"
        assert decision.base_model is None

    def test_hep_adapter_swap(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("Simulate the ATLAS detector response for Higgs decay")
        assert decision.domain == "physics"
        assert decision.is_hep is True
        assert decision.adapter == "physics_hep"

    def test_code_hep_adapter(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("Write code to run a Geant4 Monte Carlo simulation")
        assert decision.domain == "code"
        assert decision.is_hep is True
        assert decision.adapter == "code_hep"

    def test_forced_domain(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("Hello world", force_domain="physics")
        assert decision.domain == "physics"
        assert decision.domain_confidence == 1.0

    def test_auto_domain_same_as_none(self, config: JarvisConfig) -> None:
        r = Router(config)
        d1 = r.route("What is quantum mechanics?", force_domain="auto")
        d2 = r.route("What is quantum mechanics?", force_domain=None)
        assert d1.domain == d2.domain

    def test_difficulty_included(self, config: JarvisConfig) -> None:
        r = Router(config)
        decision = r.route("What is the speed of light?")
        assert decision.difficulty in ("easy", "medium", "hard")
        assert decision.difficulty_confidence > 0
