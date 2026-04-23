"""Debug script to inspect LiveCodeBench dataset and identify eval issues."""
import json
import sys

DATA_PATH = "/work/hdd/bgde/jhill5/data/benchmarks/livecode/livecode_bench.jsonl"

with open(DATA_PATH) as f:
    problems = [json.loads(l) for l in f]

print(f"Total problems: {len(problems)}")
print(f"Keys in first problem: {list(problems[0].keys())}")
print()

# Check fn_name and starter_code
has_fn = sum(1 for p in problems if p.get("fn_name"))
has_starter = sum(1 for p in problems if p.get("starter_code"))
print(f"Has fn_name: {has_fn}/{len(problems)}")
print(f"Has starter_code: {has_starter}/{len(problems)}")
print()

# Test case distribution
tests = [len(p.get("test_cases", [])) for p in problems]
print("Test case counts:")
for count in sorted(set(tests)):
    print(f"  {count} tests: {tests.count(count)} problems")
print()

# Sample test case format
for p in problems[:5]:
    tc = p.get("test_cases", [])
    if tc:
        print(f"Problem: {p.get('id', '?')} | title: {p.get('title', '?')[:50]}")
        print(f"  n_tests: {len(tc)}")
        print(f"  test[0] keys: {list(tc[0].keys()) if isinstance(tc[0], dict) else type(tc[0])}")
        sample = json.dumps(tc[0])[:200]
        print(f"  test[0]: {sample}")
        print()

# Check the last eval results for failure patterns
EVAL_PATH = "/work/hdd/bgde/jhill5/eval/livecode_20260326.json"
try:
    with open(EVAL_PATH) as f:
        results = json.load(f)
    details = results.get("details", [])

    # Count failure reasons
    reasons = {}
    for d in details:
        if d.get("passed") is None:
            reason = d.get("reason", "unknown")
        elif d.get("passed"):
            reason = "passed"
        else:
            reason = f"failed_{d.get('tests_passed', 0)}/{d.get('tests_total', 0)}"
        reasons[reason] = reasons.get(reason, 0) + 1

    print("Result breakdown:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print()

    # Show some failed examples with partial passes
    print("Examples of partial failures (some tests passed):")
    partial = [d for d in details if d.get("tests_passed", 0) > 0 and not d.get("passed")]
    for d in partial[:5]:
        print(f"  {d.get('id', '?')}: {d.get('tests_passed')}/{d.get('tests_total')} tests passed")
    print(f"  Total partial failures: {len(partial)}")

except FileNotFoundError:
    print(f"No eval results found at {EVAL_PATH}")

# Show functional test case samples
print("\n--- Functional test case samples ---")
count = 0
for p in problems:
    tc = p.get("test_cases", [])
    if tc and tc[0].get("testtype") == "functional":
        print(f"Problem: {p.get('id','?')} | {p.get('title','')[:40]}")
        print(f"  starter_code: '{p.get('starter_code','')[:100]}'")
        print(f"  test[0]: {json.dumps(tc[0])[:300]}")
        print()
        count += 1
        if count >= 3:
            break
