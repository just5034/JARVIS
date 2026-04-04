#!/usr/bin/env python3
"""Deep diagnostic: inspect actual model outputs and dataset format.

Usage (on Delta):
    python scripts/diagnose_eval_v2.py
"""

import json
import sys
from pathlib import Path

EVAL_DIR = "/scratch/bgde/jhill5/eval"
DATA_DIR = "/scratch/bgde/jhill5/data/benchmarks"


def find_latest(prefix):
    """Find the latest result file matching prefix."""
    files = sorted(Path(EVAL_DIR).glob(f"{prefix}_qwen35_*.json"))
    return files[-1] if files else None


def diagnose_livecode_responses():
    """Check what response_tail looks like for no_code_extracted problems."""
    path = find_latest("livecode")
    if not path:
        print("No livecode results found")
        return

    print("=" * 60)
    print(f"LIVECODEBENCH RESPONSE INSPECTION ({path.name})")
    print("=" * 60)

    with open(path) as f:
        r = json.load(f)

    details = r["details"]
    has_tail = [d for d in details if "response_tail" in d]
    no_code = [d for d in details if d.get("reason") == "no_code_extracted"]

    print(f"Total problems: {len(details)}")
    print(f"No code extracted: {len(no_code)}")
    print(f"Have response_tail: {len(has_tail)}")
    print()

    # Show first 5 response tails
    shown = 0
    for d in details:
        if d.get("reason") == "no_code_extracted" and "response_tail" in d:
            print(f"--- Problem: {d.get('id', '?')} / {d.get('title', '?')} ---")
            print(d["response_tail"])
            print()
            shown += 1
            if shown >= 5:
                break

    if shown == 0:
        print("WARNING: No response_tail stored. The fix may not have been picked up.")
        print("Checking if any detail has extra keys beyond standard...")
        if details:
            all_keys = set()
            for d in details[:50]:
                all_keys.update(d.keys())
            print(f"  Keys found in details: {sorted(all_keys)}")


def diagnose_dataset_format():
    """Inspect the LiveCodeBench dataset to understand unknown-difficulty problems."""
    lcb_path = Path(DATA_DIR) / "livecode" / "livecode_bench.jsonl"
    if not lcb_path.exists():
        print(f"Dataset not found at {lcb_path}")
        return

    print("=" * 60)
    print("LIVECODEBENCH DATASET INSPECTION")
    print("=" * 60)

    with open(lcb_path) as f:
        problems = [json.loads(line) for line in f]

    # Difficulty distribution
    by_diff = {}
    for p in problems:
        d = p.get("difficulty", "MISSING")
        by_diff[d] = by_diff.get(d, 0) + 1

    print(f"Total problems: {len(problems)}")
    print(f"Difficulty distribution:")
    for d, count in sorted(by_diff.items(), key=lambda kv: -kv[1]):
        print(f"  {d:20s}: {count}")
    print()

    # Check unknown problems
    unknown = [p for p in problems if p.get("difficulty", "unknown") == "unknown"]
    known = [p for p in problems if p.get("difficulty", "unknown") != "unknown"]

    print(f"Known difficulty: {len(known)}")
    print(f"Unknown difficulty: {len(unknown)}")
    print(f"Unknown with non-empty description: {sum(1 for p in unknown if p.get('description', '').strip())}")
    print(f"Unknown with test_cases: {sum(1 for p in unknown if p.get('test_cases'))}")
    print(f"Unknown with starter_code: {sum(1 for p in unknown if p.get('starter_code', '').strip())}")
    print()

    # Show 2 sample unknown problems
    for i, p in enumerate(unknown[:2]):
        print(f"--- Unknown problem #{i+1} ---")
        print(f"  id: {p.get('id')}")
        print(f"  title: {p.get('title')}")
        print(f"  difficulty: {p.get('difficulty')}")
        print(f"  description length: {len(p.get('description', ''))}")
        print(f"  starter_code: {repr(p.get('starter_code', '')[:100])}")
        print(f"  test_cases count: {len(p.get('test_cases', []))}")
        tc = p.get("test_cases", [])
        if tc and isinstance(tc[0], dict):
            print(f"  test_case[0] keys: {sorted(tc[0].keys())}")
            print(f"  test_case[0] testtype: {tc[0].get('testtype', 'MISSING')}")
        print(f"  description[:300]: {p.get('description', '')[:300]}")
        print()

    # Show 1 sample known problem for comparison
    if known:
        p = known[0]
        print(f"--- Known problem (for comparison) ---")
        print(f"  id: {p.get('id')}")
        print(f"  title: {p.get('title')}")
        print(f"  difficulty: {p.get('difficulty')}")
        print(f"  description length: {len(p.get('description', ''))}")
        print(f"  starter_code: {repr(p.get('starter_code', '')[:100])}")
        print(f"  test_cases count: {len(p.get('test_cases', []))}")
        tc = p.get("test_cases", [])
        if tc and isinstance(tc[0], dict):
            print(f"  test_case[0] keys: {sorted(tc[0].keys())}")
            print(f"  test_case[0] testtype: {tc[0].get('testtype', 'MISSING')}")
        print(f"  description[:300]: {p.get('description', '')[:300]}")
        print()


def diagnose_aime_extraction():
    """Check what the model actually put in \\boxed{} for wrong AIME answers."""
    path = find_latest("aime")
    if not path:
        print("No AIME results found")
        return

    print("=" * 60)
    print(f"AIME EXTRACTION INSPECTION ({path.name})")
    print("=" * 60)

    with open(path) as f:
        r = json.load(f)

    wrong = [d for d in r["details"] if not d["is_correct"]]
    print(f"Wrong answers: {len(wrong)}")
    print()

    # Check if responses are stored
    has_response = any("response" in d for d in r["details"])
    if has_response:
        # Search for \boxed in wrong answers
        import re
        for d in wrong:
            resp = d.get("response", "")
            boxed_matches = re.findall(r"\\boxed\{", resp)
            # Find all boxed content with brace matching
            print(f"  #{d.get('number','?')} expected={d['expected']} got={d['predicted']}")
            print(f"    \\boxed count: {len(boxed_matches)}")
            # Show last 300 chars of response
            print(f"    response tail: ...{resp[-300:]}")
            print()
    else:
        print("No response field stored in AIME results.")
        print("Need to add response logging to run_aime.py")
        print()
        print("Wrong answer pattern analysis:")
        for d in wrong:
            print(f"  #{d.get('number','?')} expected={d['expected']} got={d['predicted']}")


def main():
    diagnose_livecode_responses()
    print("\n")
    diagnose_dataset_format()
    print("\n")
    diagnose_aime_extraction()


if __name__ == "__main__":
    main()
