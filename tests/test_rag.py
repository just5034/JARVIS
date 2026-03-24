"""Tests for Phase 6: RAG knowledge base."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.rag.augmenter import PromptAugmenter
from jarvis.rag.retriever import PhysicsRetriever


@pytest.fixture
def corpus_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "physics_corpus.json"


# --- PhysicsRetriever ---


class TestPhysicsRetriever:
    def test_load_corpus(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        assert retriever.loaded
        assert retriever.corpus_size > 0

    def test_keyword_retrieval_higgs(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        results = retriever.retrieve("What is the Higgs boson mass?", top_k=3)
        assert len(results) > 0
        # Should find the Higgs passage
        assert any("125" in r for r in results)

    def test_keyword_retrieval_qcd(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        results = retriever.retrieve("strong coupling constant QCD", top_k=3)
        assert len(results) > 0
        assert any("α_s" in r or "strong" in r.lower() for r in results)

    def test_keyword_retrieval_detector(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        results = retriever.retrieve("ATLAS detector calorimeter", top_k=3)
        assert len(results) > 0
        assert any("ATLAS" in r for r in results)

    def test_empty_query(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        results = retriever.retrieve("", top_k=3)
        # Empty query should return empty or minimal results
        assert isinstance(results, list)

    def test_no_corpus(self) -> None:
        retriever = PhysicsRetriever(corpus_path=Path("/nonexistent/path.json"))
        retriever.load()
        assert not retriever.loaded
        assert retriever.retrieve("anything") == []

    def test_top_k_limit(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        results = retriever.retrieve("physics particles energy mass", top_k=2)
        assert len(results) <= 2

    def test_unit_conversions(self, corpus_path: Path) -> None:
        retriever = PhysicsRetriever(corpus_path=corpus_path)
        retriever.load()
        results = retriever.retrieve("Convert GeV to joules", top_k=3)
        assert len(results) > 0
        assert any("GeV" in r for r in results)


# --- PromptAugmenter ---


class TestPromptAugmenter:
    def test_augment_adds_system_message(self) -> None:
        augmenter = PromptAugmenter()
        messages = [{"role": "user", "content": "What is the Higgs mass?"}]
        passages = ["The Higgs boson mass is 125.25 GeV."]
        result = augmenter.augment(messages, passages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "125.25" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_augment_appends_to_existing_system(self) -> None:
        augmenter = PromptAugmenter()
        messages = [
            {"role": "system", "content": "You are a physics expert."},
            {"role": "user", "content": "What is the Higgs mass?"},
        ]
        passages = ["The Higgs boson mass is 125.25 GeV."]
        result = augmenter.augment(messages, passages)

        assert len(result) == 2
        assert "physics expert" in result[0]["content"]
        assert "125.25" in result[0]["content"]

    def test_augment_empty_passages(self) -> None:
        augmenter = PromptAugmenter()
        messages = [{"role": "user", "content": "Hello"}]
        result = augmenter.augment(messages, [])
        assert result == messages

    def test_augment_multiple_passages(self) -> None:
        augmenter = PromptAugmenter()
        messages = [{"role": "user", "content": "Tell me about particles"}]
        passages = ["Passage 1", "Passage 2", "Passage 3"]
        result = augmenter.augment(messages, passages)

        system_content = result[0]["content"]
        assert "Passage 1" in system_content
        assert "Passage 2" in system_content
        assert "Passage 3" in system_content

    def test_augment_does_not_mutate_original(self) -> None:
        augmenter = PromptAugmenter()
        original = [{"role": "user", "content": "Hello"}]
        augmenter.augment(original, ["Passage"])
        assert len(original) == 1  # Original unchanged
