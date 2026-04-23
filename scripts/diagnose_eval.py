#!/usr/bin/env python3
"""Diagnose eval failures from the latest baseline run.

Usage (on Delta):
    python scripts/diagnose_eval.py /work/hdd/bgde/jhill5/eval/livecode_qwen35_20260403_061207.json
    python scripts/diagnose_eval.py /work/hdd/bgde/jhill5/eval/aime_2024_qwen35_20260403_061207.json
"""

import json
import re
import sys
from pathlib import Path


def diagnose_livecode(results: dict) -> None:
    details = results["details"]
    metrics = results["metrics"]

    print("=" * 60)
    print("LIVECODEBENCH DIAGNOSIS")
    print("=" * 60)
    print(f"pass@1: {metrics['pass_at_1']:.1%} ({metrics['n_correct']}/{metrics['n_total']})")
    print()

    # Categorize failures
    no_code = [x for x in details if x.get("reason") == "no_code_extracted"]
    no_tests = [x for x in details if x.get("reason") == "no_test_cases"]
    passed = [x for x in details if x.get("passed") is True]
    failed_with_code = [x for x in details if x.get("passed") is False and "tests_passed" in x]
    zero_passed = [x for x in failed_with_code if x["tests_passed"] == 0]
    some_passed = [x for x in failed_with_code if x["tests_passed"] > 0]

    print("--- Failure Breakdown ---")
    print(f"  Passed:              {len(passed)}")
    print(f"  No code extracted:   {len(no_code)}")
    print(f"  No test cases:       {len(no_tests)}")
    print(f"  Code ran, 0 tests:   {len(zero_passed)}")
    print(f"  Code ran, partial:   {len(some_passed)}")
    print(f"  Total:               {len(details)}")
    print()

    # Problem type breakdown
    by_type = {}
    for x in details:
        pt = x.get("problem_type", x.get("reason", "unknown"))
        if pt not in by_type:
            by_type[pt] = {"passed": 0, "failed": 0}
        if x.get("passed") is True:
            by_type[pt]["passed"] += 1
        else:
            by_type[pt]["failed"] += 1

    print("--- By Problem Type ---")
    for pt, counts in sorted(by_type.items()):
        total = counts["passed"] + counts["failed"]
        rate = counts["passed"] / total if total > 0 else 0
        print(f"  {pt:20s}: {counts['passed']}/{total} ({rate:.1%})")
    print()

    # Difficulty breakdown
    by_diff = {}
    for x in details:
        d = x.get("difficulty", "unknown")
        if d not in by_diff:
            by_diff[d] = {"passed": 0, "total": 0}
        by_diff[d]["total"] += 1
        if x.get("passed") is True:
            by_diff[d]["passed"] += 1

    print("--- By Difficulty ---")
    for d, counts in sorted(by_diff.items()):
        rate = counts["passed"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {d:20s}: {counts['passed']}/{counts['total']} ({rate:.1%})")
    print()

    # Test pass distribution for failures
    if failed_with_code:
        pass_counts = {}
        for x in failed_with_code:
            ratio = f"{x['tests_passed']}/{x['tests_total']}"
            pass_counts[ratio] = pass_counts.get(ratio, 0) + 1

        print("--- Test Pass Distribution (failures only) ---")
        for ratio, count in sorted(pass_counts.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {ratio:10s}: {count} problems")
        print()

    # Sample failures to inspect
    print("=" * 60)
    print("SAMPLE FAILURES (first 5 zero-test-pass problems)")
    print("=" * 60)
    for x in zero_passed[:5]:
        print(f"\n--- Problem: {x.get('title', x.get('id', '?'))} ({x.get('difficulty','?')}, {x.get('problem_type','?')}) ---")
        print(f"  tests: 0/{x['tests_total']}")
        # Check if the full response is stored
        if "response" in x:
            resp = x["response"]
            # Show last 500 chars of response to see what the model output
            print(f"  response tail (500 chars):")
            print(f"    {resp[-500:]}")
        elif "code" in x:
            code = x["code"]
            print(f"  extracted code ({len(code)} chars):")
            print(f"    {code[:300]}...")
        else:
            print("  (no response/code stored in details — see below)")

    # Check if responses are stored at all
    has_response = sum(1 for x in details if "response" in x)
    has_code = sum(1 for x in details if "code" in x)
    print(f"\n--- Detail fields available ---")
    print(f"  Details with 'response' field: {has_response}/{len(details)}")
    print(f"  Details with 'code' field:     {has_code}/{len(details)}")

    if has_response == 0 and has_code == 0:
        print("\n  WARNING: No response/code stored in results JSON.")
        print("  Cannot inspect what the model actually generated.")
        print("  Re-run with response logging to debug further.")
        print("  See FIX SUGGESTIONS below.")

    # Actionable suggestions
    print()
    print("=" * 60)
    print("FIX SUGGESTIONS")
    print("=" * 60)
    if len(no_code) > 50:
        print(f"[HIGH] {len(no_code)} problems had no code extracted.")
        print("  -> Code extraction is likely broken for Qwen3.5's output format.")
        print("  -> Check if model wraps code in fences after </think>.")
        print("  -> May need to strip thinking BEFORE extracting code.")
    if len(zero_passed) > len(details) * 0.3:
        print(f"[HIGH] {len(zero_passed)} problems had code but 0 tests passed.")
        print("  -> Could be: wrong code extracted (from thinking), runtime errors,")
        print("     or test case format mismatch.")
        print("  -> Run a single problem manually to check:")
        print("     python -c \"from training.eval.run_livecode import *; ...\"")
    if len(some_passed) > 50:
        print(f"[MED] {len(some_passed)} problems passed SOME but not all tests.")
        print("  -> Model generates plausible code but has edge case bugs.")
        print("  -> This is real model performance, not a harness bug.")


def diagnose_aime(results: dict) -> None:
    details = results["details"]
    metrics = results["metrics"]

    print("=" * 60)
    print("AIME 2024 DIAGNOSIS")
    print("=" * 60)
    print(f"accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['n_total']})")
    print()

    # Show all problems with predictions
    print("--- All Problems ---")
    for i, x in enumerate(details):
        marker = "OK" if x["is_correct"] else "XX"
        pred = x.get("predicted", "None")
        exp = x.get("expected", "?")
        title = x.get("problem", "")[:80]
        contest = x.get("contest", "")
        num = x.get("number", "")
        print(f"  [{marker}] #{num} expected={exp} predicted={pred}  {title}...")
    print()

    # Extraction failures
    no_answer = [x for x in details if x.get("predicted") is None]
    wrong = [x for x in details if not x["is_correct"] and x.get("predicted") is not None]

    print("--- Failure Analysis ---")
    print(f"  Correct:        {metrics['n_correct']}")
    print(f"  No answer extracted: {len(no_answer)}")
    print(f"  Wrong answer:   {len(wrong)}")
    print()

    if no_answer:
        print("--- Problems with NO answer extracted ---")
        for x in no_answer:
            print(f"  #{x.get('number','?')} expected={x.get('expected','?')}")
            if "response" in x:
                # Show tail to see what format the answer is in
                print(f"    response tail: ...{x['response'][-200:]}")
        print()

    if wrong:
        print("--- Wrong answers ---")
        for x in wrong:
            print(f"  #{x.get('number','?')} expected={x.get('expected','?')} got={x.get('predicted','?')}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_eval.py <results.json>")
        print()
        print("Examples:")
        print("  python scripts/diagnose_eval.py /work/hdd/bgde/jhill5/eval/livecode_qwen35_20260403_061207.json")
        print("  python scripts/diagnose_eval.py /work/hdd/bgde/jhill5/eval/aime_2024_qwen35_20260403_061207.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    with open(path) as f:
        results = json.load(f)

    # Detect benchmark type from filename or content
    name = path.name.lower()
    if "livecode" in name:
        diagnose_livecode(results)
    elif "aime" in name:
        diagnose_aime(results)
    elif "gpqa" in name:
        print("GPQA passed (84.9%) — no diagnosis needed.")
    else:
        print(f"Unknown benchmark type for {path.name}")
        print("Dumping metrics:")
        print(json.dumps(results.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
