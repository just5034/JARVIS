"""ARIA Prototype v2 — Adaptive Reasoning with Iterative Accumulation

Tests the hypothesis: Can N informed passes (with verified intermediate results
cached between passes) beat N independent passes (standard best-of-N)?

Built on validated Qwen3.5 eval methodology:
- Thinking mode ON by default (<think>...</think> then answer)
- Sampling: temp=0.6, top_p=0.95, top_k=20 (published Qwen3.5 defaults)
- Answer extraction: strip_thinking() + extract_boxed_answer() from eval base
- Verify/extract steps use lower max_tokens to avoid wasting thinking budget

Usage:
  # Against vLLM serving Qwen3.5-27B (the real test)
  python scripts/aria_prototype.py \\
      --backend openai \\
      --base-url http://localhost:8192/v1 \\
      --model /projects/bgde/jhill5/models/qwen3.5-27b \\
      --max-passes 3

  # Against Anthropic (quick iteration)
  python scripts/aria_prototype.py --backend anthropic --api-key sk-ant-...

  # Custom problem set
  python scripts/aria_prototype.py --backend openai --problems path/to/problems.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3.5 output handling — ported from training/eval/base.py
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Extract the final answer portion from model output with thinking.

    Qwen3.5 outputs: "<think>...reasoning...</think>final answer"
    Returns text after </think>, or full text if no thinking block found.

    IMPORTANT: Qwen3.5 often puts useful content (answers, code) INSIDE the
    thinking block. Callers should search both stripped and full text.
    """
    if "</think>" in text:
        _, _, after = text.rpartition("</think>")
        after = after.strip()
        if after:
            return after
    return text


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the LAST \boxed{...} answer from model output.

    Handles nested braces like \boxed{3\cdot 5^{2}}.
    Ported from training/eval/base.py — this is the validated extractor.
    """
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[start : j - 1].strip())
        i = j if depth == 0 else idx + 1

    if matches:
        return matches[-1]
    return None


def extract_numeric(text: str) -> str | None:
    r"""Extract numeric answer from model output (AIME-style: integer 0-999).

    Search order: \boxed{} → "the answer is X" → trailing "= X" → last number.
    Ported from training/eval/base.py.
    """
    boxed = extract_boxed_answer(text)
    if boxed:
        s = boxed.strip().strip("$").strip()
        if re.fullmatch(r"-?\d+", s):
            return s
        no_comma = s.replace(",", "")
        if re.fullmatch(r"-?\d+", no_comma):
            return no_comma
        nums = re.findall(r"-?\d+", s)
        if nums:
            return nums[-1]

    match = re.search(r"[Tt]he\s+answer\s+is\s*[:\s]*(-?\d+)", text)
    if match:
        return match.group(1)

    match = re.search(r"=\s*(-?\d+)\s*$", text.strip())
    if match:
        return match.group(1)

    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        return matches[-1]

    return None


def extract_answer_robust(text: str) -> str | None:
    """Extract answer searching both post-thinking and full text.

    Qwen3.5 may place the \\boxed{} answer inside <think> or after </think>.
    This matches the eval harness behavior: try stripped first, fall back to full.
    """
    stripped = strip_thinking(text)
    answer = extract_numeric(stripped)
    if answer is not None:
        return answer
    # Fall back: search full output (answer may be inside thinking block)
    return extract_numeric(text)


# ---------------------------------------------------------------------------
# Qwen3.5 sampling defaults (from training/eval/base.py)
# ---------------------------------------------------------------------------

QWEN35_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    # top_k=20 is Qwen3.5 published default, but OpenAI API doesn't support it.
    # vLLM's OpenAI-compat server may support it via extra_body.
}


# ---------------------------------------------------------------------------
# LLM Backend Abstraction
# ---------------------------------------------------------------------------

class LLMBackend:
    """Thin wrapper for Anthropic / OpenAI-compatible backends."""

    def __init__(self, backend: str, model: str, api_key: str | None, base_url: str | None):
        self.backend = backend
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if backend == "anthropic":
            import anthropic
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            self.client = anthropic.Anthropic(**kwargs)
        elif backend == "openai":
            from openai import OpenAI
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate(
        self,
        messages: list[dict],
        temperature: float = QWEN35_DEFAULTS["temperature"],
        top_p: float = QWEN35_DEFAULTS["top_p"],
        max_tokens: int = 30000,
    ) -> str:
        if self.backend == "anthropic":
            system = None
            chat_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    chat_msgs.append(m)

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": chat_msgs,
            }
            if system:
                kwargs["system"] = system

            resp = self.client.messages.create(**kwargs)
            self.total_input_tokens += resp.usage.input_tokens
            self.total_output_tokens += resp.usage.output_tokens
            return resp.content[0].text

        elif self.backend == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if resp.usage:
                self.total_input_tokens += resp.usage.prompt_tokens
                self.total_output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Problem Set — AIME-style problems with known answers
# ---------------------------------------------------------------------------

BUILTIN_PROBLEMS = [
    {
        "id": "aime_2024_i_1",
        "problem": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s km/hr, the walk takes her 4 minutes more than if she walks at s+2 km/hr. Find the value of s^2 + s.",
        "answer": "99",
    },
    {
        "id": "aime_2024_i_3",
        "problem": "Alice and Bob play a game. They alternate turns, with Alice going first. On Alice's turn, she picks any integer from 1 to 10 inclusive and adds it to a running total. On Bob's turn, he picks any integer from 1 to 10 inclusive and adds it to the running total. The player who causes the running total to reach or exceed 100 wins. What is the smallest value of the running total on Alice's first turn such that Alice has a winning strategy?",
        "answer": "1",
    },
    {
        "id": "aime_2024_i_5",
        "problem": "Rectangles ABCD and EFGH are drawn such that D, E, C, F are collinear. Also, A, D, H, G all lie on a circle. If BC = 16, AB = 107, FG = 17, and EF = 184, what is the length of CE?",
        "answer": "104",
    },
    {
        "id": "aime_2024_i_9",
        "problem": "Let the sequence of rationals a_1, a_2, a_3, ... be defined by a_1 = 2025 and for k >= 1, a_{k+1} = a_k if a_k is an integer, and a_{k+1} = a_k + 1/(floor(a_k)) otherwise. Find the value of a_{2025}. Express as an integer.",
        "answer": "59",
    },
    {
        "id": "custom_integral_1",
        "problem": "Compute the definite integral of (x^2 * e^x) from 0 to 1. Express your answer in the form ae - b where a and b are positive integers. What is a + b?",
        "answer": "3",
    },
    {
        "id": "logic_1",
        "problem": "Five people (A, B, C, D, E) sit in a row. A refuses to sit next to B. C must sit next to D. How many valid seating arrangements are there?",
        "answer": "36",
    },
]


# ---------------------------------------------------------------------------
# Reasoning Cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    sub_problem: str
    result: str
    confidence: str  # "high", "medium", "low"
    method: str
    pass_number: int


@dataclass
class Challenge:
    """A specific concern raised by the adversarial verifier."""
    concern: str
    reason: str
    alternative_approach: str
    pass_number: int


@dataclass
class ReasoningCache:
    """v3 cache: tracks challenges (skeptic-prior) and verified facts (only when adversary concedes)."""
    verified_facts: list[CacheEntry] = field(default_factory=list)
    challenges: list[Challenge] = field(default_factory=list)
    previous_answers: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)  # SUSPECT or SOLID per pass

    def format_challenges_section(self) -> str:
        """Format challenges for injection into the retry solve prompt."""
        if not self.challenges:
            return "(no specific concerns raised, but the answer is still suspect)"

        parts = []
        for i, c in enumerate(self.challenges, 1):
            parts.append(f"{i}. {c.concern}")
            if c.reason:
                parts.append(f"   Reason: {c.reason}")
            if c.alternative_approach and c.alternative_approach.lower() not in ("none", "none if confident"):
                parts.append(f"   Suggested alternative: {c.alternative_approach}")
        return "\n".join(parts)

    def previous_answers_str(self) -> str:
        if not self.previous_answers:
            return "(none)"
        return ", ".join(self.previous_answers)


# ---------------------------------------------------------------------------
# Prompts — v3: ADVERSARIAL VERIFICATION
#
# v2 negative result: same-model self-verification has confirmation bias.
# When the solver is wrong, the verifier (same model) "verifies" the wrong
# answer, populating the cache with bad scaffolding for subsequent passes.
#
# v3 design: assume the answer is WRONG until proven otherwise.
# - Verifier is told the answer is suspected incorrect
# - Verifier must produce a SPECIFIC challenge (concrete error or alternative)
# - Cache only stores facts when adversary explicitly concedes "no errors found"
# - Subsequent solve passes are told "the previous answer X was CHALLENGED
#   on these grounds: [...]. Try a different approach."
# ---------------------------------------------------------------------------

# Solve prompt — v3 takes an explicit "previous attempt" section instead of
# trusting cached "verified facts" (which were the v2 confirmation-bias trap).
SOLVE_PROMPT_INITIAL = """Solve the following competition math problem. The answer is an integer.

Problem: {problem}

Please reason step by step, and put your final answer within \\boxed{{}}."""


SOLVE_PROMPT_RETRY = """Solve the following competition math problem. The answer is an integer.

## PREVIOUS ATTEMPT WAS CHALLENGED
Previous answer(s) attempted: {previous_answers}

A skeptical reviewer found the following specific concerns:
{challenges}

You should treat the previous answer(s) as SUSPECT. The reviewer's challenges may or may not be correct, but the consistent failure suggests a systematic error. Try a fundamentally different approach:
- Re-read the problem carefully — are you interpreting it correctly?
- Use a different solution method (if you used algebra, try cases; if you used cases, try a closed form)
- Check edge cases and boundary conditions
- Verify any assumptions you're making

Problem: {problem}

Please reason step by step, and put your final answer within \\boxed{{}}."""


# ADVERSARIAL verification — the key v3 change.
# Frames the verifier as a hostile reviewer who assumes the answer is wrong.
# This breaks the confirmation bias loop where solver and verifier agree
# because they share the same blind spots.
ADVERSARIAL_VERIFY_PROMPT = """You are a SKEPTICAL REVIEWER. Your job is to find errors in this solution. The proposed answer is suspected to be INCORRECT — your task is to find the specific mistake.

Problem: {problem}

Proposed answer: {answer}

Solution to critique:
{solution_stripped}

Instructions for your review:
1. Re-read the problem statement carefully. Is the solution interpreting it correctly? Common errors: misreading what's asked, ignoring constraints, off-by-one indexing.
2. For each major step, ask "could this be wrong?" — find at least one plausible error.
3. Try to construct a counterexample or alternative answer.
4. Only conclude the answer is correct if you've genuinely tried and failed to find any error.

After your analysis, output your findings in EXACTLY this format (after your thinking):

CHALLENGES:
- [specific concern about a step or assumption] | [why it might be wrong]

ALTERNATIVE_APPROACH: [a different method that might give a different answer, or "none if confident"]

VERDICT: [SUSPECT if you found plausible errors, SOLID if you genuinely could not find any error after trying]

VERIFIED_FACTS: [intermediate results you actually checked and confirmed correct, in format "fact = value | method"; or "none" if VERDICT is SUSPECT]"""


# ---------------------------------------------------------------------------
# Cache extraction — parse the structured output from verify step
# ---------------------------------------------------------------------------

def parse_adversarial_verification(
    raw_output: str, pass_number: int
) -> tuple[list[Challenge], list[CacheEntry], str, str]:
    """Parse adversarial verifier output.

    Returns (challenges, verified_facts, alternative_approach, verdict).

    verdict is 'SUSPECT', 'SOLID', or 'unknown'.
    Always searches the full raw output — Qwen3.5 puts structured content
    inside <think> blocks, so post-</think> text is often empty/summary-only.
    """
    # ALWAYS search full output. Qwen3.5 puts structured markers inside <think>.
    # Searching only post-</think> was the v3 bug: 29/30 passes returned "unknown".
    text = raw_output

    # Log what we're parsing for debug
    stripped = strip_thinking(raw_output)
    has_markers_in_think = "CHALLENGES:" in raw_output or "VERDICT:" in raw_output
    has_markers_post_think = "CHALLENGES:" in stripped or "VERDICT:" in stripped
    logger.debug(
        f"Verify output: {len(raw_output)} chars, "
        f"markers_in_full={has_markers_in_think}, markers_post_think={has_markers_post_think}"
    )

    challenges = []
    facts = []
    alternative = ""
    verdict = "unknown"

    in_challenges = False
    in_verified_facts = False

    # Flexible marker detection — Qwen3.5 may wrap markers in markdown
    # formatting like **CHALLENGES:**, ## CHALLENGES, etc.
    def _match_marker(line: str, marker: str) -> tuple[bool, str]:
        """Check if line contains a section marker, return (matched, remainder)."""
        # Strip markdown formatting
        clean = line.strip().strip("*#").strip()
        if clean.upper().startswith(marker.upper()):
            remainder = clean[len(marker):].strip().strip(":").strip()
            return True, remainder
        return False, ""

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        matched, remainder = _match_marker(line, "CHALLENGES:")
        if not matched:
            matched, remainder = _match_marker(line, "CHALLENGES")
        if matched:
            in_challenges = True
            in_verified_facts = False
            if remainder and remainder.lower() != "none":
                _parse_challenge_line(remainder, challenges, alternative, pass_number)
            i += 1
            continue

        matched, remainder = _match_marker(line, "ALTERNATIVE_APPROACH:")
        if not matched:
            matched, remainder = _match_marker(line, "ALTERNATIVE APPROACH:")
        if not matched:
            matched, remainder = _match_marker(line, "ALTERNATIVE_APPROACH")
        if matched:
            in_challenges = False
            in_verified_facts = False
            alternative = remainder
            i += 1
            continue

        matched, remainder = _match_marker(line, "VERDICT:")
        if not matched:
            matched, remainder = _match_marker(line, "VERDICT")
        if matched:
            in_challenges = False
            in_verified_facts = False
            v = remainder.upper()
            if "SUSPECT" in v:
                verdict = "SUSPECT"
            elif "SOLID" in v:
                verdict = "SOLID"
            i += 1
            continue

        matched, remainder = _match_marker(line, "VERIFIED_FACTS:")
        if not matched:
            matched, remainder = _match_marker(line, "VERIFIED FACTS:")
        if not matched:
            matched, remainder = _match_marker(line, "VERIFIED_FACTS")
        if matched:
            in_challenges = False
            in_verified_facts = True
            if remainder and remainder.lower() != "none":
                _parse_fact_line(remainder, facts, pass_number)
            i += 1
            continue

        if in_challenges and (line.startswith("- ") or line.startswith("* ")):
            _parse_challenge_line(line.lstrip("-* "), challenges, alternative, pass_number)
        elif in_verified_facts and (line.startswith("- ") or line.startswith("* ")):
            _parse_fact_line(line.lstrip("-* "), facts, pass_number)

        i += 1

    # Skeptic prior: only trust verified_facts if verdict is SOLID
    if verdict != "SOLID":
        facts = []

    # Log parse results for debug
    if verdict == "unknown" and not challenges:
        # Log a snippet so we can diagnose WHY parsing failed
        snippet = raw_output[-500:] if len(raw_output) > 500 else raw_output
        logger.warning(
            f"Parse returned unknown/empty. Last 500 chars of verify output:\n{snippet}"
        )

    # Attach alternative approach to challenges (it's the adversary's main suggestion)
    if alternative and challenges:
        for c in challenges:
            if not c.alternative_approach:
                c.alternative_approach = alternative

    return challenges, facts, alternative, verdict


def _parse_challenge_line(line: str, challenges: list[Challenge], alternative: str, pass_number: int):
    """Parse a CHALLENGE line: '[concern] | [reason]'."""
    parts = line.split("|")
    concern = parts[0].strip().strip("- ")
    reason = parts[1].strip() if len(parts) > 1 else ""
    if concern:
        challenges.append(Challenge(
            concern=concern,
            reason=reason,
            alternative_approach=alternative,
            pass_number=pass_number,
        ))


def _parse_fact_line(line: str, facts: list[CacheEntry], pass_number: int):
    """Parse a VERIFIED_FACTS line: '[description] = [value] | [method]'."""
    parts = line.split("|")
    fact_text = parts[0].strip()
    method = parts[1].strip() if len(parts) > 1 else ""

    if "=" in fact_text:
        sub_prob, result = fact_text.rsplit("=", 1)
        facts.append(CacheEntry(
            sub_problem=sub_prob.strip().strip("- "),
            result=result.strip(),
            confidence="high",
            method=method,
            pass_number=pass_number,
        ))
    elif fact_text:
        facts.append(CacheEntry(
            sub_problem=fact_text.strip("- "),
            result="(verified)",
            confidence="high",
            method=method,
            pass_number=pass_number,
        ))


# ---------------------------------------------------------------------------
# ARIA: Multi-Pass Reasoning Engine
# ---------------------------------------------------------------------------

def run_aria(
    llm: LLMBackend,
    problem: str,
    max_passes: int = 2,
    solve_max_tokens: int = 30000,
    verify_max_tokens: int = 8192,
    verbose: bool = True,
) -> dict:
    """Run ARIA v3 — adversarial multi-pass reasoning.

    Flow:
    1. Solve (initial prompt)
    2. Adversarial verify — verifier assumes answer is wrong, tries to break it
    3. If verdict == SOLID: early exit (no point retrying a confirmed answer)
    4. Else: solve again with the specific challenges injected, retry
    5. Final answer is the LAST pass that produced SOLID, OR a vote if no pass was SOLID

    Token budget:
    - SOLVE: Full budget (~30K) — extended thinking
    - VERIFY: 8K — Qwen3.5 needs ~3-5K for thinking before structured output
    """
    cache = ReasoningCache()
    all_attempts = []  # list of (answer, verdict) tuples
    all_raw_solutions = []
    early_exit_pass = None
    start_time = time.time()

    for pass_num in range(1, max_passes + 1):
        if verbose:
            logger.info(f"  ARIA Pass {pass_num}/{max_passes}")

        # --- SOLVE ---
        if pass_num == 1 or not cache.challenges:
            solve_prompt = SOLVE_PROMPT_INITIAL.format(problem=problem)
        else:
            solve_prompt = SOLVE_PROMPT_RETRY.format(
                problem=problem,
                previous_answers=cache.previous_answers_str(),
                challenges=cache.format_challenges_section(),
            )

        solution_raw = llm.generate(
            [{"role": "user", "content": solve_prompt}],
            max_tokens=solve_max_tokens,
        )
        all_raw_solutions.append(solution_raw)

        answer = extract_answer_robust(solution_raw)
        if answer:
            cache.previous_answers.append(answer)

        if verbose:
            logger.info(f"    Pass {pass_num} answer: {answer}")

        # --- ADVERSARIAL VERIFY ---
        solution_stripped = strip_thinking(solution_raw)
        if len(solution_stripped) < 100:
            solution_stripped = solution_raw[-3000:]

        verify_prompt = ADVERSARIAL_VERIFY_PROMPT.format(
            problem=problem,
            answer=answer or "unknown",
            solution_stripped=solution_stripped[:3000],
        )

        verify_raw = llm.generate(
            [{"role": "user", "content": verify_prompt}],
            temperature=0.2,
            max_tokens=verify_max_tokens,
        )

        new_challenges, new_facts, alternative, verdict = parse_adversarial_verification(
            verify_raw, pass_num,
        )
        cache.challenges.extend(new_challenges)
        cache.verified_facts.extend(new_facts)
        cache.verdicts.append(verdict)

        all_attempts.append((answer, verdict))

        if verbose:
            logger.info(
                f"    Verdict: {verdict} | Challenges: {len(new_challenges)} | "
                f"Verified facts: {len(new_facts)} (cumulative: {len(cache.verified_facts)})"
            )

        # --- EARLY EXIT on SOLID verdict ---
        # If the adversarial verifier genuinely could not find an error,
        # there's no point running more passes — the answer is as confident
        # as same-model verification can make it.
        if verdict == "SOLID":
            if verbose:
                logger.info(f"    Early exit: SOLID verdict at pass {pass_num}")
            early_exit_pass = pass_num
            break

    elapsed = time.time() - start_time

    # --- FINAL ANSWER SELECTION ---
    # Priority:
    # 1. If any pass got SOLID verdict, use the LAST SOLID answer
    # 2. Otherwise, majority vote across all attempts (with last as tiebreaker)
    solid_answers = [ans for ans, v in all_attempts if v == "SOLID" and ans is not None]
    if solid_answers:
        final_answer = solid_answers[-1]
        selection_method = "solid_verdict"
    else:
        all_ans = [ans for ans, _ in all_attempts if ans is not None]
        if all_ans:
            from collections import Counter
            counts = Counter(all_ans)
            top = counts.most_common(1)[0]
            # If there's a tie, use the last attempt
            if top[1] > 1:
                final_answer = top[0]
                selection_method = "majority_vote"
            else:
                final_answer = all_ans[-1]
                selection_method = "last_attempt_no_majority"
        else:
            final_answer = None
            selection_method = "no_answer"

    return {
        "method": "aria_v3_adversarial",
        "passes_run": len(all_attempts),
        "passes_max": max_passes,
        "early_exit_pass": early_exit_pass,
        "final_answer": final_answer,
        "selection_method": selection_method,
        "all_answers": [a for a, _ in all_attempts if a is not None],
        "verdicts": cache.verdicts,
        "cache_facts": len(cache.verified_facts),
        "cache_challenges": len(cache.challenges),
        "elapsed_seconds": elapsed,
        "total_input_tokens": llm.total_input_tokens,
        "total_output_tokens": llm.total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Baseline: Independent Passes (standard best-of-N)
# ---------------------------------------------------------------------------

def run_baseline(
    llm: LLMBackend,
    problem: str,
    num_passes: int = 2,
    solve_max_tokens: int = 30000,
    verbose: bool = True,
) -> dict:
    """Run N independent attempts (no cache sharing) and pick majority answer."""
    answers = []
    start_time = time.time()

    for pass_num in range(1, num_passes + 1):
        if verbose:
            logger.info(f"  Baseline Pass {pass_num}/{num_passes}")

        solve_prompt = SOLVE_PROMPT_INITIAL.format(problem=problem)
        solution_raw = llm.generate(
            [{"role": "user", "content": solve_prompt}],
            max_tokens=solve_max_tokens,
        )

        answer = extract_answer_robust(solution_raw)
        if answer:
            answers.append(answer)

        if verbose:
            logger.info(f"    Pass {pass_num} answer: {answer}")

    elapsed = time.time() - start_time

    # Majority vote
    if answers:
        from collections import Counter
        vote = Counter(answers).most_common(1)[0][0]
    else:
        vote = None

    return {
        "method": "baseline",
        "passes": num_passes,
        "final_answer": vote,
        "all_answers": answers,
        "elapsed_seconds": elapsed,
        "total_input_tokens": llm.total_input_tokens,
        "total_output_tokens": llm.total_output_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARIA v3 — Adversarial Multi-Pass Reasoning (Qwen3.5-aware)")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: anthropic=claude-haiku-4-5-20251001, openai uses --base-url model")
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoints")
    parser.add_argument("--problems", default=None, help="Path to JSON file with problems")
    parser.add_argument("--max-passes", type=int, default=2, help="Number of passes")
    parser.add_argument("--solve-max-tokens", type=int, default=30000,
                        help="Max tokens for solve step (default: 30000, leaves room for prompt)")
    parser.add_argument("--verify-max-tokens", type=int, default=8192,
                        help="Max tokens for verify step (default: 8192, Qwen3.5 needs ~5K thinking + structured output)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False
        logging.getLogger().setLevel(logging.WARNING)

    if args.model is None:
        if args.backend == "anthropic":
            args.model = "claude-haiku-4-5-20251001"
        else:
            args.model = "default"

    if args.problems:
        with open(args.problems) as f:
            problems = json.load(f)
    else:
        problems = BUILTIN_PROBLEMS

    logger.info(f"Backend: {args.backend}, Model: {args.model}")
    logger.info(f"Problems: {len(problems)}, Passes: {args.max_passes}")
    logger.info(f"Solve tokens: {args.solve_max_tokens}, Verify tokens: {args.verify_max_tokens}")
    logger.info(f"Sampling: temp={QWEN35_DEFAULTS['temperature']}, top_p={QWEN35_DEFAULTS['top_p']}")
    logger.info("=" * 70)

    results = []

    for prob in problems:
        pid = prob["id"]
        problem_text = prob["problem"]
        correct = str(prob["answer"]).strip()

        logger.info(f"\n{'='*70}")
        logger.info(f"Problem: {pid}")
        logger.info(f"Expected answer: {correct}")
        logger.info("-" * 70)

        # --- Baseline ---
        logger.info(f"[BASELINE] Running {args.max_passes} independent passes...")
        baseline_llm = LLMBackend(args.backend, args.model, args.api_key, args.base_url)
        baseline_result = run_baseline(
            baseline_llm, problem_text, args.max_passes,
            args.solve_max_tokens, args.verbose,
        )
        baseline_correct = str(baseline_result["final_answer"]).strip() == correct

        logger.info(f"[BASELINE] Final: {baseline_result['final_answer']} | Correct: {baseline_correct}")
        logger.info(f"[BASELINE] Tokens: {baseline_result['total_input_tokens']}in + {baseline_result['total_output_tokens']}out | Time: {baseline_result['elapsed_seconds']:.1f}s")

        # --- ARIA v3 (adversarial) ---
        logger.info(f"\n[ARIA] Running adversarial multi-pass (max {args.max_passes} passes, early-exit on SOLID)...")
        aria_llm = LLMBackend(args.backend, args.model, args.api_key, args.base_url)
        aria_result = run_aria(
            aria_llm, problem_text, args.max_passes,
            args.solve_max_tokens, args.verify_max_tokens, args.verbose,
        )
        aria_correct = str(aria_result["final_answer"]).strip() == correct

        logger.info(f"[ARIA] Final: {aria_result['final_answer']} | Correct: {aria_correct}")
        logger.info(f"[ARIA] Selection: {aria_result['selection_method']} | Verdicts: {aria_result['verdicts']}")
        logger.info(f"[ARIA] Passes run: {aria_result['passes_run']}/{aria_result['passes_max']} (early exit at: {aria_result['early_exit_pass']})")
        logger.info(f"[ARIA] Cache: {aria_result['cache_facts']} verified facts, {aria_result['cache_challenges']} challenges")
        logger.info(f"[ARIA] Tokens: {aria_result['total_input_tokens']}in + {aria_result['total_output_tokens']}out | Time: {aria_result['elapsed_seconds']:.1f}s")

        results.append({
            "problem_id": pid,
            "correct_answer": correct,
            "baseline": {**baseline_result, "is_correct": baseline_correct},
            "aria": {**aria_result, "is_correct": aria_correct},
        })

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_correct_count = sum(1 for r in results if r["baseline"]["is_correct"])
    aria_correct_count = sum(1 for r in results if r["aria"]["is_correct"])
    total = len(results)

    baseline_tokens = sum(r["baseline"]["total_input_tokens"] + r["baseline"]["total_output_tokens"] for r in results)
    aria_tokens = sum(r["aria"]["total_input_tokens"] + r["aria"]["total_output_tokens"] for r in results)

    print(f"\n{'Method':<20} {'Correct':<12} {'Accuracy':<12} {'Total Tokens':<15}")
    print("-" * 60)
    print(f"{'Baseline (indep.)':<20} {baseline_correct_count}/{total:<11} {baseline_correct_count/total*100:.1f}%{'':<7} {baseline_tokens:<15}")
    print(f"{'ARIA (informed)':<20} {aria_correct_count}/{total:<11} {aria_correct_count/total*100:.1f}%{'':<7} {aria_tokens:<15}")

    if baseline_tokens > 0:
        print(f"\nToken overhead: ARIA uses {aria_tokens/baseline_tokens:.2f}x tokens vs baseline")

    print(f"\n{'Problem':<25} {'Baseline':<18} {'ARIA':<18} {'Winner':<10}")
    print("-" * 72)
    for r in results:
        pid = r["problem_id"][:24]
        ba = r["baseline"]["final_answer"]
        aa = r["aria"]["final_answer"]
        b = "CORRECT" if r["baseline"]["is_correct"] else f"WRONG ({ba})"
        a = "CORRECT" if r["aria"]["is_correct"] else f"WRONG ({aa})"
        if r["aria"]["is_correct"] and not r["baseline"]["is_correct"]:
            winner = "ARIA"
        elif r["baseline"]["is_correct"] and not r["aria"]["is_correct"]:
            winner = "Baseline"
        elif r["aria"]["is_correct"] and r["baseline"]["is_correct"]:
            winner = "Tie"
        else:
            winner = "Neither"
        print(f"{pid:<25} {b:<18} {a:<18} {winner:<10}")

    # Save
    output_path = args.output or f"scripts/aria_results_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "backend": args.backend,
                "model": args.model,
                "max_passes": args.max_passes,
                "solve_max_tokens": args.solve_max_tokens,
                "verify_max_tokens": args.verify_max_tokens,
                "sampling": QWEN35_DEFAULTS,
                "num_problems": total,
            },
            "summary": {
                "baseline_accuracy": baseline_correct_count / total if total else 0,
                "aria_accuracy": aria_correct_count / total if total else 0,
                "baseline_total_tokens": baseline_tokens,
                "aria_total_tokens": aria_tokens,
                "token_ratio": aria_tokens / max(baseline_tokens, 1),
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
