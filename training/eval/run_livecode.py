"""Evaluate model on LiveCodeBench — code generation with execution.

Dataset: livecodebench/code_generation_lite
Metric: pass@1 (% of problems where generated code passes all test cases)
Target: >= 65% for code brain

Usage:
    python -m training.eval.run_livecode \
        --model /projects/bgde/jhill5/models/qwen2.5-coder-32b-instruct \
        --adapter /projects/bgde/jhill5/adapters/code_general \
        --output /scratch/bgde/jhill5/eval/livecode.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from training.eval.base import (
    generate_batch,
    load_model,
    make_arg_parser,
)


LIVECODE_PROMPT_TEMPLATE = """Solve the following programming problem. Write a complete Python solution.

{problem_description}

{input_format}

{output_format}

{constraints}

Write your solution as a complete Python program that reads from stdin and writes to stdout. Put your code inside a ```python code block."""


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
            "input_format": row.get("input_format", ""),
            "output_format": row.get("output_format", ""),
            "constraints": row.get("constraints", ""),
        }

        # Test cases may be in different formats
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


def extract_python_code(text: str) -> str | None:
    """Extract Python code from model output (looks for code blocks)."""
    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: look for code-like content (def/import/class at start of line)
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r"^(import |from |def |class |if |for |while |print)", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return None


def run_code_safe(code: str, stdin: str, timeout: int = 30) -> tuple[bool, str]:
    """Run Python code in a subprocess with timeout.

    Returns:
        (success, stdout_or_error)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
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


def check_test_cases(code: str, test_cases: list[dict], timeout: int = 30) -> tuple[int, int]:
    """Run code against test cases. Returns (passed, total)."""
    if not test_cases:
        return 0, 0

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        test_input = tc.get("input", "")
        expected = tc.get("output", tc.get("expected_output", "")).strip()

        success, output = run_code_safe(code, test_input, timeout)
        if success and output.strip() == expected:
            passed += 1

    return passed, total


def evaluate(args) -> dict:
    """Run LiveCodeBench evaluation."""
    problems = load_livecode(args.data_dir)
    print(f"[livecode] loaded {len(problems)} problems")

    llm = load_model(args.model, args.adapter)

    # Build prompts
    prompts = []
    for p in problems:
        prompt = LIVECODE_PROMPT_TEMPLATE.format(
            problem_description=p["description"],
            input_format=f"Input format: {p['input_format']}" if p["input_format"] else "",
            output_format=f"Output format: {p['output_format']}" if p["output_format"] else "",
            constraints=f"Constraints: {p['constraints']}" if p["constraints"] else "",
        )
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
        response_text = responses[0]
        code = extract_python_code(response_text)

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
            details.append({
                "id": problem["id"],
                "title": problem.get("title", ""),
                "passed": False,
                "reason": "no_code_extracted",
            })
            continue

        passed, total = check_test_cases(code, test_cases)
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
