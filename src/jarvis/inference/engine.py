"""Inference engine — difficulty-aware strategy selection and execution.

Orchestrates candidate sampling, voting, verification, and budget forcing
based on the difficulty level determined by the router.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.config import InferenceConfig
from jarvis.inference.budget_forcing import BudgetForcer
from jarvis.inference.sampling import CandidateSampler
from jarvis.inference.verification import ThinkPRMVerifier
from jarvis.inference.voting import SelfConsistencyVoter
from jarvis.rag.augmenter import PromptAugmenter
from jarvis.rag.retriever import PhysicsRetriever

if TYPE_CHECKING:
    from jarvis.brains.model_loader import LoadedModelHandle

logger = logging.getLogger(__name__)

# Domain-specific verification chain prompts (appended to user message for medium/hard)
_VERIFICATION_PROMPTS = {
    "math": "\n\nAfter solving, verify your answer by substituting back into the original equation.",
    "physics": "\n\nAfter deriving your result, check dimensional analysis and verify limiting cases.",
    "code": "\n\nAfter writing your solution, test it mentally with edge cases including empty input, single element, and large values.",
    "chemistry": "\n\nAfter your analysis, verify conservation of mass and charge balance.",
    "general": "\n\nDouble-check your reasoning step by step before giving your final answer.",
}


@dataclass
class InferenceResult:
    """Result from the inference engine."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    strategy: str
    num_candidates: int
    verification_score: float | None = None
    budget_forcing_rounds: int = 0


class InferenceEngine:
    """Selects and runs the inference strategy based on difficulty level."""

    def __init__(
        self,
        config: InferenceConfig,
        retriever: PhysicsRetriever | None = None,
    ) -> None:
        self.config = config
        self._sampler = CandidateSampler()
        self._voter = SelfConsistencyVoter()
        self._verifier = ThinkPRMVerifier()
        self._budget_forcer = BudgetForcer()
        self._verifier_loaded = False
        self._retriever = retriever
        self._augmenter = PromptAugmenter()

    def _ensure_verifier(self) -> None:
        """Lazy-load the verifier on first hard query."""
        if not self._verifier_loaded:
            self._verifier.load()
            self._verifier_loaded = True

    def _apply_verification_chain(
        self, messages: list[dict], domain: str
    ) -> list[dict]:
        """Append a verification prompt to the last user message."""
        suffix = _VERIFICATION_PROMPTS.get(domain, _VERIFICATION_PROMPTS["general"])
        messages = [m.copy() for m in messages]

        # Find last user message and append verification prompt
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = {
                    "role": "user",
                    "content": messages[i]["content"] + suffix,
                }
                break

        return messages

    async def generate(
        self,
        model: "LoadedModelHandle",
        messages: list[dict],
        difficulty: str,
        domain: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> InferenceResult:
        """Run difficulty-aware inference strategy."""
        level_config = self.config.difficulty_levels.get(difficulty)
        if level_config is None:
            level_config = self.config.difficulty_levels.get("easy")
            if level_config is None:
                raise ValueError(f"No config for difficulty '{difficulty}' or 'easy'")

        strategy = level_config.strategy

        # Apply RAG augmentation for physics queries
        gen_messages = messages
        if domain == "physics" and self._retriever and self._retriever.loaded:
            user_text = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    user_text = m["content"]
                    break
            if user_text:
                passages = self._retriever.retrieve(user_text, top_k=5)
                if passages:
                    gen_messages = self._augmenter.augment(gen_messages, passages)
                    logger.info("RAG: augmented physics query with %d passages", len(passages))

        # Apply verification chain for medium/hard
        if level_config.verification_chain:
            gen_messages = self._apply_verification_chain(messages, domain)

        if strategy == "single_pass":
            return await self._single_pass(
                model, gen_messages, temperature, top_p, max_tokens, stop
            )
        elif strategy == "best_of_n":
            return await self._best_of_n(
                model, gen_messages, domain, level_config.num_candidates,
                top_p, max_tokens, stop,
            )
        elif strategy == "best_of_n_verified":
            return await self._best_of_n_verified(
                model, gen_messages, domain, level_config,
                top_p, max_tokens, stop,
            )
        else:
            logger.warning("Unknown strategy '%s', falling back to single_pass", strategy)
            return await self._single_pass(
                model, gen_messages, temperature, top_p, max_tokens, stop
            )

    async def _single_pass(
        self, model, messages, temperature, top_p, max_tokens, stop,
    ) -> InferenceResult:
        """Easy: single forward pass."""
        results = await self._sampler.sample(
            model, messages, n=1,
            temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, stop=stop,
        )
        r = results[0]
        return InferenceResult(
            text=r.text,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            finish_reason=r.finish_reason,
            strategy="single_pass",
            num_candidates=1,
        )

    async def _best_of_n(
        self, model, messages, domain, n, top_p, max_tokens, stop,
    ) -> InferenceResult:
        """Medium: generate N candidates, vote for best."""
        results = await self._sampler.sample(
            model, messages, n=n,
            temperature=0.7,  # Need diversity for voting
            top_p=top_p, max_tokens=max_tokens, stop=stop,
        )

        candidates = [r.text for r in results]
        winner_text, confidence = self._voter.vote(candidates, domain)

        # Find the matching result for token counts
        winner_result = results[0]
        for r in results:
            if r.text == winner_text:
                winner_result = r
                break

        return InferenceResult(
            text=winner_text,
            prompt_tokens=winner_result.prompt_tokens,
            completion_tokens=winner_result.completion_tokens,
            finish_reason=winner_result.finish_reason,
            strategy="best_of_n",
            num_candidates=n,
        )

    async def _best_of_n_verified(
        self, model, messages, domain, level_config, top_p, max_tokens, stop,
    ) -> InferenceResult:
        """Hard: generate N candidates, vote, verify with ThinkPRM, budget force."""
        n = level_config.num_candidates

        # Step 1: Generate N candidates
        results = await self._sampler.sample(
            model, messages, n=n,
            temperature=0.7, top_p=top_p,
            max_tokens=max_tokens, stop=stop,
        )

        candidates = [r.text for r in results]

        # Step 2: Self-consistency voting to narrow the field
        winner_text, vote_confidence = self._voter.vote(candidates, domain)

        # Step 3: ThinkPRM verification on top candidates
        self._ensure_verifier()
        verification_score = None

        if self._verifier.available:
            # Get unique top candidates (up to 4) for verification
            seen = set()
            top_candidates = []
            for c in candidates:
                normalized = c.strip()
                if normalized not in seen:
                    seen.add(normalized)
                    top_candidates.append(c)
                    if len(top_candidates) >= 4:
                        break

            pessimistic = level_config.selection == "pessimistic"
            winner_text, verification_score = self._verifier.select_best(
                top_candidates, pessimistic=pessimistic
            )
        else:
            verification_score = None

        # Step 4: Budget forcing on the winner
        budget_forcing_rounds = 0
        if level_config.budget_forcing:
            max_waits = level_config.budget_forcing_max_waits or 3
            self._budget_forcer = BudgetForcer(max_waits=max_waits)
            self._budget_forcer.reset()

            # Find the result matching the winner for token count
            winner_completion_tokens = 0
            for r in results:
                if r.text == winner_text:
                    winner_completion_tokens = r.completion_tokens
                    break

            thinking_budget = level_config.thinking_budget_tokens

            while self._budget_forcer.should_force(
                winner_text, winner_completion_tokens, thinking_budget
            ):
                # Re-generate with the partial output as context
                extended_messages = messages.copy()
                extended_messages.append({"role": "assistant", "content": winner_text})

                continuation_prompt = self._budget_forcer.apply(winner_text)
                extended_messages.append({"role": "user", "content": continuation_prompt})

                continuation_results = await self._sampler.sample(
                    model, extended_messages, n=1,
                    temperature=0.3,  # Lower temp for refinement
                    top_p=top_p, max_tokens=max_tokens, stop=stop,
                )

                if continuation_results:
                    winner_text = winner_text + continuation_results[0].text
                    winner_completion_tokens += continuation_results[0].completion_tokens

            budget_forcing_rounds = self._budget_forcer.force_count

        # Find matching result for token counts
        winner_result = results[0]
        for r in results:
            if r.text in winner_text:
                winner_result = r
                break

        return InferenceResult(
            text=winner_text,
            prompt_tokens=winner_result.prompt_tokens,
            completion_tokens=winner_result.completion_tokens,
            finish_reason=winner_result.finish_reason,
            strategy="best_of_n_verified",
            num_candidates=n,
            verification_score=verification_score,
            budget_forcing_rounds=budget_forcing_rounds,
        )
