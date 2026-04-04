"""Evaluate model on LiveCodeBench — code generation with execution.

Dataset: bzantium/livecodebench (mirror of livecodebench/code_generation_lite)
Metric: pass@1 (% of problems where generated code passes all test cases)
Target: >= 65% for code brain

Eval methodology aligned with official LiveCodeBench harness:
- Separate prompts for stdin/stdout vs function-call (LeetCode) problems
- Extract LAST code block from model output (not first)
- Numeric output comparison with Decimal fallback
- Pre-import common modules in execution environment

Usage:
    python -m training.eval.run_livecode \
        --model /projects/bgde/jhill5/models/qwen2.5-coder-32b-instruct \
        --output /scratch/bgde/jhill5/eval/livecode.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from training.eval.base import (
    generate_batch,
    load_model,
    make_arg_parser,
    strip_thinking,
)

# Matches base.py strip_thinking — used to try post-think extraction first
_strip_thinking = strip_thinking


# Prompt for stdin/stdout problems (Codeforces/AtCoder style)
STDIN_PROMPT = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

{problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.
```python
# YOUR CODE HERE
```"""

# Prompt for function-call problems (LeetCode style with starter code)
FUNCTION_PROMPT = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

{problem_description}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters as follows.
```python
{starter_code}
```"""


# Pre-import block injected into execution environment (matches official harness)
PREIMPORT_BLOCK = """\
import sys
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import itertools
import functools
import operator
from collections import defaultdict, Counter, deque
from itertools import permutations, combinations, accumulate
from functools import lru_cache, reduce
from typing import List, Optional, Tuple, Dict, Set
sys.setrecursionlimit(50000)
"""


def load_livecode(data_dir: str) -> list[dict]:
    """Load LiveCodeBench problems from local cache or HuggingFace."""
    local_path = Path(data_dir) / "livecode" / "livecode_bench.jsonl"

    if local_path.exists():
        print(f"[livecode] loading from {local_path}")
        with open(local_path) as f:
            return [json.loads(line) for line in f]

    print("[livecode] local data not found, downloading from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset("bzantium/livecodebench", "release_v5", split="test")
    except Exception as e:
        print(f"ERROR: could not load LiveCodeBench: {e}", file=sys.stderr)
        print("Run 'python -m training.data.download_benchmarks' first.", file=sys.stderr)
        sys.exit(1)

    problems = []
    for row in ds:
        problem = {
            "id": row.get("question_id", row.get("id", len(problems))),
            "title": row.get("question_title", row.get("title", "")),
            "description": row.get("question_content", row.get("description", "")),
            "difficulty": row.get("difficulty", "unknown"),
            "starter_code": row.get("starter_code", ""),
            "fn_name": row.get("fn_name", None),
            "input_format": row.get("input_format", ""),
            "output_format": row.get("output_format", ""),
            "constraints": row.get("constraints", ""),
        }

        if "public_test_cases" in row:
            test_data = row["public_test_cases"]
            if isinstance(test_data, str):
                try:
                    test_data = json.loads(test_data)
                except json.JSONDecodeError:
                    test_data = []
            problem["test_cases"] = test_data
        elif "test_cases" in row:
            problem["test_cases"] = row["test_cases"]
        else:
            problem["test_cases"] = []

        problems.append(problem)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"[livecode] cached {len(problems)} problems to {local_path}")

    return problems


def _extract_from_fences(text: str) -> str | None:
    """Extract the LAST fenced code block from text."""
    lines = text.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if "```" in line]

    if len(fence_indices) >= 2:
        start = fence_indices[-2] + 1
        end = fence_indices[-1]
        code = "\n".join(lines[start:end]).strip()
        if code:
            return code
    return None


def _extract_from_heuristic(text: str) -> str | None:
    """Fallback: look for code-like content."""
    code_lines = []
    in_code = False
    for line in text.split("\n"):
        if re.match(r"^(import |from |def |class |if |for |while |print|sys\.)", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()
    return None


def extract_python_code(text: str) -> str | None:
    """Extract the LAST code block from model output.

    Strategy: try the post-thinking text first (after </think>),
    then fall back to the full output. This prevents grabbing
    exploratory code from Qwen3.5's thinking section.
    """
    # 1. Try post-thinking text first (where the final answer should be)
    after_think = _strip_thinking(text)
    if after_think != text:  # thinking was actually stripped
        code = _extract_from_fences(after_think)
        if code:
            return code

    # 2. Fall back to full text (covers models without thinking)
    code = _extract_from_fences(text)
    if code:
        return code

    # 3. Heuristic fallback on post-thinking text
    if after_think != text:
        code = _extract_from_heuristic(after_think)
        if code:
            return code

    # 4. Heuristic fallback on full text
    return _extract_from_heuristic(text)


def get_problem_type(problem: dict) -> str:
    """Determine if a problem is stdin or functional based on test case metadata."""
    test_cases = problem.get("test_cases", [])
    if test_cases and isinstance(test_cases[0], dict):
        return test_cases[0].get("testtype", "stdin")
    return "stdin"


def extract_fn_name(starter_code: str) -> str | None:
    """Extract the method name from LeetCode-style starter code.

    Example: 'class Solution:\\n    def countSeniors(self, ...' -> 'countSeniors'
    """
    match = re.search(r"def\s+(\w+)\s*\(\s*self", starter_code)
    if match:
        return match.group(1)
    return None


def compare_outputs(actual: str, expected: str) -> bool:
    """Compare outputs with Decimal numeric fallback (matches official harness)."""
    actual_lines = actual.strip().splitlines()
    expected_lines = expected.strip().splitlines()

    if len(actual_lines) != len(expected_lines):
        return False

    for a_line, e_line in zip(actual_lines, expected_lines):
        a = a_line.strip()
        e = e_line.strip()

        # Exact match first
        if a == e:
            continue

        # Decimal numeric fallback
        try:
            if Decimal(a) == Decimal(e):
                continue
        except (InvalidOperation, ValueError):
            pass

        return False

    return True


def run_code_safe(code: str, stdin: str, timeout: int = 30) -> tuple[bool, str]:
    """Run Python code in a subprocess with timeout and pre-imports."""
    full_code = PREIMPORT_BLOCK + "\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr[:500]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)[:500]
        finally:
            Path(f.name).unlink(missing_ok=True)


def run_function_call(code: str, fn_name: str, test_input: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a function-call style problem (LeetCode) by calling the function directly.

    Test inputs may be multi-line, where each line is a separate argument:
      e.g., "[12, 9]\\n1" -> two args: [12, 9] and 1
    """
    # Parse each line as a separate JSON argument
    args = []
    for line in test_input.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            args.append(line)

    args_repr = ", ".join(repr(a) for a in args)
    wrapper = f"""{PREIMPORT_BLOCK}
import json

{code}

_sol = Solution()
_result = _sol.{fn_name}({args_repr})
if isinstance(_result, tuple):
    _result = list(_result)
print(json.dumps(_result))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr[:500]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)[:500]
        finally:
            Path(f.name).unlink(missing_ok=True)


def check_test_cases(
    code: str,
    test_cases: list[dict],
    fn_name: str | None = None,
    timeout: int = 30,
) -> tuple[int, int]:
    """Run code against test cases. Returns (passed, total)."""
    if not test_cases:
        return 0, 0

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        test_input = tc.get("input", "")
        expected = tc.get("output", tc.get("expected_output", "")).strip()

        if fn_name:
            # Function-call problem: call Solution().fn_name() with parsed args
            success, output = run_function_call(code, fn_name, test_input, timeout)
            if success:
                # Compare JSON outputs for function-call problems
                try:
                    actual_val = json.loads(output.strip())
                    expected_val = json.loads(expected)
                    # Convert tuples to lists
                    if isinstance(actual_val, tuple):
                        actual_val = list(actual_val)
                    if isinstance(expected_val, tuple):
                        expected_val = list(expected_val)
                    if actual_val == expected_val:
                        passed += 1
                except (json.JSONDecodeError, TypeError):
                    # Fall back to string comparison
                    if compare_outputs(output, expected):
                        passed += 1
        else:
            # Stdin/stdout problem
            success, output = run_code_safe(code, test_input, timeout)
            if success and compare_outputs(output, expected):
                passed += 1

    return passed, total


def evaluate(args) -> dict:
    """Run LiveCodeBench evaluation."""
    problems = load_livecode(args.data_dir)
    print(f"[livecode] loaded {len(problems)} problems")

    # Count problem types using testtype field
    n_stdin = sum(1 for p in problems if get_problem_type(p) == "stdin")
    n_func = sum(1 for p in problems if get_problem_type(p) == "functional")
    print(f"[livecode] {n_stdin} stdin/stdout problems, {n_func} function-call problems")

    llm = load_model(args.model, args.adapter)

    # Build prompts — different templates for stdin vs function-call problems
    prompts = []
    for p in problems:
        ptype = get_problem_type(p)
        starter = p.get("starter_code", "")
        if ptype == "functional" and starter:
            prompt = FUNCTION_PROMPT.format(
                problem_description=p["description"],
                starter_code=starter,
            )
        else:
            prompt = STDIN_PROMPT.format(problem_description=p["description"])
        prompts.append(prompt)

    # Generate
    all_responses = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        responses = generate_batch(
            llm,
            batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            adapter_path=args.adapter,
        )
        all_responses.extend(responses)
        print(f"[livecode] generated {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    # Score via execution
    correct = 0
    total_with_tests = 0
    details = []
    by_difficulty: dict[str, dict] = {}

    for problem, responses in zip(problems, all_responses):
        full_text = responses[0]
        code = extract_python_code(full_text)

        test_cases = problem.get("test_cases", [])
        if not test_cases:
            details.append({
                "id": problem["id"],
                "title": problem.get("title", ""),
                "passed": None,
                "reason": "no_test_cases",
            })
            continue

        total_with_tests += 1

        if code is None:
            # Store tail of response for debugging extraction failures
            details.append({
                "id": problem["id"],
                "title": problem.get("title", ""),
                "passed": False,
                "reason": "no_code_extracted",
                "response_tail": full_text[-500:] if full_text else "",
            })
            continue

        # Determine problem type and function name
        ptype = get_problem_type(problem)
        fn_name = None
        if ptype == "functional":
            fn_name = extract_fn_name(problem.get("starter_code", ""))

        passed, total = check_test_cases(code, test_cases, fn_name=fn_name)
        is_correct = passed == total and total > 0

        if is_correct:
            correct += 1

        diff = problem.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        if is_correct:
            by_difficulty[diff]["correct"] += 1

        details.append({
            "id": problem["id"],
            "title": problem.get("title", ""),
            "passed": is_correct,
            "tests_passed": passed,
            "tests_total": total,
            "difficulty": diff,
            "problem_type": "function_call" if fn_name else "stdin",
        })

    pass_at_1 = correct / total_with_tests if total_with_tests > 0 else 0
    metrics = {
        "pass_at_1": round(pass_at_1, 4),
        "n_correct": correct,
        "n_total": total_with_tests,
        "n_skipped_no_tests": len(problems) - total_with_tests,
    }

    for diff, counts in sorted(by_difficulty.items()):
        if counts["total"] > 0:
            metrics[f"pass_at_1_{diff}"] = round(counts["correct"] / counts["total"], 4)

    return {"metrics": metrics, "details": details}


def main():
    parser = make_arg_parser("livecode_bench")
    args = parser.parse_args()

    results = evaluate(args)

    from training.utils.tracking import create_run, log_eval_results

    tracker = None
    if not args.no_track:
        tracker = create_run(
            experiment=args.experiment,
            hparams={"model": args.model, "adapter": args.adapter},
            log_dir=args.log_dir,
        )

    log_eval_results(
        tracker,
        "livecode_bench",
        results["metrics"],
        details=results["details"],
        output_path=Path(args.output),
    )
    if tracker:
        tracker.close()

    target = 0.65
    p1 = results["metrics"]["pass_at_1"]
    status = "PASS" if p1 >= target else "FAIL"
    print(f"\n[livecode] pass@1: {p1:.1%} (target: {target:.0%}) — {status}")


if __name__ == "__main__":
    main()
