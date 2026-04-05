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
        max_tokens: int = 32768,
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
class FailureEntry:
    approach: str
    reason: str
    pass_number: int


@dataclass
class ReasoningCache:
    verified_facts: list[CacheEntry] = field(default_factory=list)
    failures: list[FailureEntry] = field(default_factory=list)
    previous_answers: list[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        if not self.verified_facts and not self.failures:
            return ""

        parts = []

        if self.verified_facts:
            parts.append("## VERIFIED INTERMEDIATE RESULTS (treat as established facts)")
            for i, entry in enumerate(self.verified_facts, 1):
                parts.append(f"{i}. [{entry.confidence} confidence] {entry.sub_problem}: {entry.result}")
                if entry.method:
                    parts.append(f"   Method: {entry.method}")

        if self.failures:
            parts.append("\n## APPROACHES THAT FAILED (do not repeat these)")
            for i, entry in enumerate(self.failures, 1):
                parts.append(f"{i}. {entry.approach}")
                parts.append(f"   Why it failed: {entry.reason}")

        if self.previous_answers:
            parts.append(f"\n## PREVIOUS ANSWERS ATTEMPTED: {', '.join(self.previous_answers)}")
            parts.append("If your work leads to one of these same answers, carefully re-verify before committing.")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts — designed for Qwen3.5 thinking mode
#
# Key design decisions:
# - SOLVE uses the same prompt format as run_aime.py (proven to work)
# - VERIFY+EXTRACT are combined into one call (fewer tokens, less parsing)
# - Structured output is requested AFTER </think> using explicit instruction
# - Token budgets are differentiated: solve=32K, verify=4K
# ---------------------------------------------------------------------------

# Matches AIME prompt from run_aime.py
SOLVE_PROMPT = """Solve the following competition math problem. The answer is an integer.

{cache_section}Problem: {problem}

Please reason step by step, and put your final answer within \\boxed{{}}."""


# Combined verify + extract in one call.
# Asks model to think about verification, then output structured results
# after </think>. Lower token budget since we only need the structured part.
VERIFY_AND_EXTRACT_PROMPT = """You are verifying a mathematical solution. Analyze each step of the reasoning for correctness.

Problem: {problem}

Proposed solution (answer was {answer}):
{solution_stripped}

After your analysis, output your findings in EXACTLY this format (after your thinking):

VERIFIED:
- [description of verified intermediate result] = [value] | [method used]

ERRORS:
- [description of error] | [why it's wrong]

CONFIDENCE: [HIGH if answer is likely correct, LOW if errors found]

If there are no verified results, write "VERIFIED: none".
If there are no errors, write "ERRORS: none"."""


# ---------------------------------------------------------------------------
# Cache extraction — parse the structured output from verify step
# ---------------------------------------------------------------------------

def parse_verification(raw_output: str, pass_number: int) -> tuple[list[CacheEntry], list[FailureEntry], str]:
    """Parse the verify+extract output into cache entries.

    Searches both post-</think> text and full output for the structured format.
    Returns (facts, failures, confidence).
    """
    # Try post-thinking first, then full text
    text = strip_thinking(raw_output)
    if "VERIFIED:" not in text and "VERIFIED:" in raw_output:
        text = raw_output

    facts = []
    failures = []
    confidence = "unknown"

    in_verified = False
    in_errors = False

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("VERIFIED:"):
            remainder = line[len("VERIFIED:"):].strip()
            if remainder.lower() == "none":
                in_verified = False
            else:
                in_verified = True
                in_errors = False
                # Check if there's content on the same line
                if remainder and remainder.lower() != "none":
                    _parse_fact_line(remainder, facts, pass_number)
            continue

        if line.startswith("ERRORS:"):
            remainder = line[len("ERRORS:"):].strip()
            if remainder.lower() == "none":
                in_errors = False
            else:
                in_errors = True
                in_verified = False
                if remainder and remainder.lower() != "none":
                    _parse_failure_line(remainder, failures, pass_number)
            continue

        if line.startswith("CONFIDENCE:"):
            confidence = line[len("CONFIDENCE:"):].strip().lower()
            in_verified = False
            in_errors = False
            continue

        if in_verified and line.startswith("- "):
            _parse_fact_line(line[2:], facts, pass_number)
        elif in_errors and line.startswith("- "):
            _parse_failure_line(line[2:], failures, pass_number)

    return facts, failures, confidence


def _parse_fact_line(line: str, facts: list[CacheEntry], pass_number: int):
    """Parse a single VERIFIED fact line."""
    # Expected: "[description] = [value] | [method]"
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


def _parse_failure_line(line: str, failures: list[FailureEntry], pass_number: int):
    """Parse a single ERRORS failure line."""
    # Expected: "[description] | [reason]"
    parts = line.split("|")
    approach = parts[0].strip().strip("- ")
    reason = parts[1].strip() if len(parts) > 1 else "error detected"
    if approach:
        failures.append(FailureEntry(
            approach=approach,
            reason=reason,
            pass_number=pass_number,
        ))


# ---------------------------------------------------------------------------
# ARIA: Multi-Pass Reasoning Engine
# ---------------------------------------------------------------------------

def run_aria(
    llm: LLMBackend,
    problem: str,
    max_passes: int = 2,
    solve_max_tokens: int = 32768,
    verify_max_tokens: int = 4096,
    verbose: bool = True,
) -> dict:
    """Run ARIA multi-pass reasoning on a single problem.

    Token budget strategy:
    - SOLVE: Full budget (32K) — model needs space for extended thinking
    - VERIFY+EXTRACT: Reduced budget (4K) — structured output is short
    """
    cache = ReasoningCache()
    all_attempts = []
    all_raw_solutions = []
    start_time = time.time()

    for pass_num in range(1, max_passes + 1):
        if verbose:
            logger.info(f"  ARIA Pass {pass_num}/{max_passes}")

        # --- SOLVE ---
        cache_section = cache.to_prompt_section()
        if cache_section:
            cache_section = f"{cache_section}\n\n"

        solve_prompt = SOLVE_PROMPT.format(cache_section=cache_section, problem=problem)

        solution_raw = llm.generate(
            [{"role": "user", "content": solve_prompt}],
            max_tokens=solve_max_tokens,
        )
        all_raw_solutions.append(solution_raw)

        # Extract answer using validated Qwen3.5 extraction pipeline
        answer = extract_answer_robust(solution_raw)
        all_attempts.append(answer)
        if answer:
            cache.previous_answers.append(answer)

        if verbose:
            logger.info(f"    Pass {pass_num} answer: {answer}")

        # --- VERIFY + EXTRACT (combined, lower token budget) ---
        # Strip thinking for a shorter solution summary to verify
        solution_stripped = strip_thinking(solution_raw)
        # If stripped is too short, use a tail of the full output
        if len(solution_stripped) < 100:
            solution_stripped = solution_raw[-3000:]

        verify_prompt = VERIFY_AND_EXTRACT_PROMPT.format(
            problem=problem,
            answer=answer or "unknown",
            solution_stripped=solution_stripped[:3000],  # Cap input length
        )

        verify_raw = llm.generate(
            [{"role": "user", "content": verify_prompt}],
            temperature=0.2,  # Low temp for verification — more deterministic
            max_tokens=verify_max_tokens,
        )

        # Parse structured output from verification
        new_facts, new_failures, confidence = parse_verification(verify_raw, pass_num)
        cache.verified_facts.extend(new_facts)
        cache.failures.extend(new_failures)

        if verbose:
            logger.info(f"    Cache: {len(cache.verified_facts)} facts, {len(cache.failures)} failures (confidence: {confidence})")

    elapsed = time.time() - start_time
    final_answer = all_attempts[-1] if all_attempts else None

    return {
        "method": "aria",
        "passes": max_passes,
        "final_answer": final_answer,
        "all_answers": [a for a in all_attempts if a is not None],
        "cache_facts": len(cache.verified_facts),
        "cache_failures": len(cache.failures),
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
    solve_max_tokens: int = 32768,
    verbose: bool = True,
) -> dict:
    """Run N independent attempts (no cache sharing) and pick majority answer."""
    answers = []
    start_time = time.time()

    for pass_num in range(1, num_passes + 1):
        if verbose:
            logger.info(f"  Baseline Pass {pass_num}/{num_passes}")

        solve_prompt = SOLVE_PROMPT.format(cache_section="", problem=problem)
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
    parser = argparse.ArgumentParser(description="ARIA v2 — Multi-Pass Reasoning (Qwen3.5-aware)")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: anthropic=claude-haiku-4-5-20251001, openai uses --base-url model")
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoints")
    parser.add_argument("--problems", default=None, help="Path to JSON file with problems")
    parser.add_argument("--max-passes", type=int, default=2, help="Number of passes")
    parser.add_argument("--solve-max-tokens", type=int, default=32768,
                        help="Max tokens for solve step (default: 32768, Qwen3.5 published)")
    parser.add_argument("--verify-max-tokens", type=int, default=4096,
                        help="Max tokens for verify step (default: 4096, lower = faster)")
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

        # --- ARIA ---
        logger.info(f"\n[ARIA] Running {args.max_passes} informed passes with reasoning cache...")
        aria_llm = LLMBackend(args.backend, args.model, args.api_key, args.base_url)
        aria_result = run_aria(
            aria_llm, problem_text, args.max_passes,
            args.solve_max_tokens, args.verify_max_tokens, args.verbose,
        )
        aria_correct = str(aria_result["final_answer"]).strip() == correct

        logger.info(f"[ARIA] Final: {aria_result['final_answer']} | Correct: {aria_correct}")
        logger.info(f"[ARIA] Cache: {aria_result['cache_facts']} facts, {aria_result['cache_failures']} failures")
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
