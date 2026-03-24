"""Tests for the Phase 3 inference amplification system."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import InferenceConfig, load_config
from jarvis.inference.budget_forcing import BudgetForcer
from jarvis.inference.context_manager import ContextManager
from jarvis.inference.engine import InferenceEngine, InferenceResult
from jarvis.inference.sampling import CandidateSampler
from jarvis.inference.verification import ThinkPRMVerifier
from jarvis.inference.voting import AnswerExtractor, SelfConsistencyVoter
from tests.conftest import MockModelHandle


@pytest.fixture
def inference_config() -> InferenceConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    config = load_config(config_dir)
    return config.inference


# --- AnswerExtractor ---


class TestAnswerExtractor:
    def test_math_boxed(self) -> None:
        ext = AnswerExtractor()
        text = "We compute... therefore \\boxed{42}"
        assert ext.extract(text, "math") == "42"

    def test_math_multiple_boxed(self) -> None:
        ext = AnswerExtractor()
        text = "First \\boxed{10}, then \\boxed{42}"
        assert ext.extract(text, "math") == "42"  # last one

    def test_math_answer_pattern(self) -> None:
        ext = AnswerExtractor()
        text = "After calculation, the answer is 3.14"
        assert "3.14" in ext.extract(text, "math")

    def test_code_fenced_block(self) -> None:
        ext = AnswerExtractor()
        text = "Here's the solution:\n```python\ndef foo(): return 1\n```"
        assert "def foo(): return 1" in ext.extract(text, "code")

    def test_code_multiple_blocks(self) -> None:
        ext = AnswerExtractor()
        text = "```python\nfirst()\n```\nThen:\n```python\nsecond()\n```"
        assert "second()" in ext.extract(text, "code")

    def test_general_normalization(self) -> None:
        ext = AnswerExtractor()
        text = "  The answer   is   here  "
        result = ext.extract(text, "general")
        assert "the answer is here" in result

    def test_physics_uses_general(self) -> None:
        ext = AnswerExtractor()
        text = "The mass is 125 GeV"
        result = ext.extract(text, "physics")
        assert "125 gev" in result


# --- SelfConsistencyVoter ---


class TestSelfConsistencyVoter:
    def test_majority_vote(self) -> None:
        voter = SelfConsistencyVoter()
        candidates = [
            "The answer is \\boxed{42}",
            "Therefore \\boxed{42}",
            "I think \\boxed{7}",
        ]
        winner, confidence = voter.vote(candidates, "math")
        assert "42" in winner
        assert confidence == pytest.approx(2 / 3)

    def test_single_candidate(self) -> None:
        voter = SelfConsistencyVoter()
        winner, confidence = voter.vote(["Only one"], "general")
        assert winner == "Only one"
        assert confidence == 1.0

    def test_no_candidates_raises(self) -> None:
        voter = SelfConsistencyVoter()
        with pytest.raises(ValueError):
            voter.vote([], "general")

    def test_all_same(self) -> None:
        voter = SelfConsistencyVoter()
        candidates = ["Same answer"] * 4
        winner, confidence = voter.vote(candidates, "general")
        assert confidence == 1.0

    def test_code_voting(self) -> None:
        voter = SelfConsistencyVoter()
        candidates = [
            "```python\ndef sort(x): return sorted(x)\n```",
            "```python\ndef sort(x): return sorted(x)\n```",
            "```python\ndef sort(x): return list(reversed(x))\n```",
        ]
        winner, confidence = voter.vote(candidates, "code")
        assert "sorted(x)" in winner
        assert confidence == pytest.approx(2 / 3)


# --- BudgetForcer ---


class TestBudgetForcer:
    def test_should_force_on_premature_conclusion(self) -> None:
        bf = BudgetForcer(max_waits=3)
        assert bf.should_force("The answer is 42", thinking_tokens=100, budget=1000)

    def test_should_not_force_when_budget_used(self) -> None:
        bf = BudgetForcer(max_waits=3)
        assert not bf.should_force("The answer is 42", thinking_tokens=600, budget=1000)

    def test_should_not_force_at_max_waits(self) -> None:
        bf = BudgetForcer(max_waits=2)
        bf._force_count = 2
        assert not bf.should_force("The answer is 42", thinking_tokens=100, budget=1000)

    def test_should_not_force_without_conclusion(self) -> None:
        bf = BudgetForcer(max_waits=3)
        assert not bf.should_force("Hmm, let me think more about", thinking_tokens=100, budget=1000)

    def test_should_force_on_boxed(self) -> None:
        bf = BudgetForcer(max_waits=3)
        assert bf.should_force("Therefore \\boxed{42}", thinking_tokens=50, budget=1000)

    def test_should_force_on_think_tag(self) -> None:
        bf = BudgetForcer(max_waits=3)
        assert bf.should_force("Done thinking</think>", thinking_tokens=50, budget=1000)

    def test_apply_appends_continuation(self) -> None:
        bf = BudgetForcer(max_waits=3)
        result = bf.apply("The answer is 42")
        assert "Wait" in result
        assert "reconsider" in result
        assert bf.force_count == 1

    def test_apply_strips_think_tag(self) -> None:
        bf = BudgetForcer(max_waits=3)
        result = bf.apply("My reasoning</think>")
        assert "</think>" not in result

    def test_reset(self) -> None:
        bf = BudgetForcer(max_waits=3)
        bf.apply("test")
        bf.apply("test")
        assert bf.force_count == 2
        bf.reset()
        assert bf.force_count == 0

    def test_zero_budget(self) -> None:
        bf = BudgetForcer(max_waits=3)
        assert not bf.should_force("The answer is 42", thinking_tokens=0, budget=0)


# --- ThinkPRMVerifier ---


class TestThinkPRMVerifier:
    def test_not_available_by_default(self) -> None:
        v = ThinkPRMVerifier()
        assert not v.available

    def test_score_returns_neutral_when_unavailable(self) -> None:
        v = ThinkPRMVerifier()
        assert v.score("some reasoning") == 0.5

    def test_select_best_single_candidate(self) -> None:
        v = ThinkPRMVerifier()
        text, score = v.select_best(["only one"])
        assert text == "only one"
        assert score == 0.5

    def test_select_best_no_candidates_raises(self) -> None:
        v = ThinkPRMVerifier()
        with pytest.raises(ValueError):
            v.select_best([])

    def test_select_best_returns_first_when_all_neutral(self) -> None:
        v = ThinkPRMVerifier()
        # All scores are 0.5 when unavailable, so first is returned
        text, score = v.select_best(["a", "b", "c"])
        assert text in ("a", "b", "c")
        assert score == 0.5


# --- ContextManager ---


class TestContextManager:
    def test_easy_config(self, inference_config: InferenceConfig) -> None:
        cm = ContextManager(inference_config)
        config = cm.get_kv_config("easy", num_candidates=1)
        assert config["kv_cache_dtype"] == "fp8"
        assert config["enable_prefix_caching"] is False  # n=1

    def test_hard_config(self, inference_config: InferenceConfig) -> None:
        cm = ContextManager(inference_config)
        config = cm.get_kv_config("hard", num_candidates=16)
        assert config["kv_cache_dtype"] == "fp8"
        assert config["kv_quant_bits"] == 2
        assert config["enable_prefix_caching"] is True

    def test_memory_estimation(self, inference_config: InferenceConfig) -> None:
        cm = ContextManager(inference_config)
        mem = cm.estimate_kv_memory_gb(32768, num_candidates=1, kv_dtype="fp8")
        assert mem > 0
        # Should be roughly 4 GB for 32K context at FP8 with 32B model
        assert 1.0 < mem < 10.0

    def test_memory_scales_with_candidates(self, inference_config: InferenceConfig) -> None:
        cm = ContextManager(inference_config)
        mem_1 = cm.estimate_kv_memory_gb(32768, num_candidates=1, kv_dtype="fp8")
        mem_4 = cm.estimate_kv_memory_gb(32768, num_candidates=4, kv_dtype="fp8")
        assert mem_4 == pytest.approx(mem_1 * 4)


# --- CandidateSampler ---


class TestCandidateSampler:
    @pytest.mark.asyncio
    async def test_sample_single(self) -> None:
        sampler = CandidateSampler()
        model = MockModelHandle(["Response A"])
        results = await sampler.sample(model, [{"role": "user", "content": "Hi"}], n=1)
        assert len(results) == 1
        assert results[0].text == "Response A"

    @pytest.mark.asyncio
    async def test_sample_multiple(self) -> None:
        sampler = CandidateSampler()
        model = MockModelHandle(["A", "B", "C"])
        results = await sampler.sample(model, [{"role": "user", "content": "Hi"}], n=3)
        assert len(results) == 3


# --- InferenceEngine ---


class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_single_pass(self, inference_config: InferenceConfig) -> None:
        engine = InferenceEngine(inference_config)
        model = MockModelHandle(["The speed of light is 3e8 m/s"])
        result = await engine.generate(
            model, [{"role": "user", "content": "What is c?"}],
            difficulty="easy", domain="physics",
        )
        assert isinstance(result, InferenceResult)
        assert result.strategy == "single_pass"
        assert result.num_candidates == 1
        assert "3e8" in result.text

    @pytest.mark.asyncio
    async def test_best_of_n(self, inference_config: InferenceConfig) -> None:
        engine = InferenceEngine(inference_config)
        model = MockModelHandle(["Answer A", "Answer A", "Answer B", "Answer A"])
        result = await engine.generate(
            model, [{"role": "user", "content": "Solve this"}],
            difficulty="medium", domain="math",
        )
        assert result.strategy == "best_of_n"
        assert result.num_candidates == 4
        assert "Answer A" in result.text

    @pytest.mark.asyncio
    async def test_best_of_n_verified(self, inference_config: InferenceConfig) -> None:
        engine = InferenceEngine(inference_config)
        responses = ["Response"] * 16
        model = MockModelHandle(responses)
        result = await engine.generate(
            model, [{"role": "user", "content": "Derive from first principles"}],
            difficulty="hard", domain="physics",
        )
        assert result.strategy == "best_of_n_verified"
        assert result.num_candidates == 16

    @pytest.mark.asyncio
    async def test_unknown_difficulty_fallback(self, inference_config: InferenceConfig) -> None:
        engine = InferenceEngine(inference_config)
        model = MockModelHandle(["Hello"])
        result = await engine.generate(
            model, [{"role": "user", "content": "Hi"}],
            difficulty="unknown", domain="general",
        )
        # Should fall back to easy/single_pass
        assert result.strategy == "single_pass"

    @pytest.mark.asyncio
    async def test_verification_chain_appended(self, inference_config: InferenceConfig) -> None:
        engine = InferenceEngine(inference_config)
        # Capture what messages the model receives
        received_messages = []

        class SpyModel(MockModelHandle):
            def generate(self, request):
                if request.messages:
                    received_messages.extend(request.messages)
                return super().generate(request)

        model = SpyModel(["Answer"] * 4)
        await engine.generate(
            model, [{"role": "user", "content": "Solve x^2=4"}],
            difficulty="medium", domain="math",
        )
        # The user message should have verification suffix
        user_msgs = [m for m in received_messages if m["role"] == "user"]
        assert any("verify" in m["content"].lower() or "substitut" in m["content"].lower() for m in user_msgs)
