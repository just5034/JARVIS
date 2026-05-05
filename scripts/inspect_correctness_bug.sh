#!/usr/bin/env bash
# Investigate why only 22/~1552 ground-truth-bearing physics traces passed
# the correctness check. Two suspected bugs:
#   (A) `answer` field is empty/missing for most traces (so they fall into
#       "no ground truth" and are kept blindly).
#   (B) Of traces that DO have `answer`, the extract_answer/normalize logic
#       fails to match (likely a string-format mismatch).
#
# Read-only. Login-node safe. No SUs.
#
# Usage:  bash scripts/inspect_correctness_bug.sh

set -uo pipefail

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
TRACES="/work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl"
PROBLEMS="/work/hdd/bgde/jhill5/data/physics_problems.jsonl"

module load python/3.13.5-gcc13.3.1 >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$HOME/JARVIS"

python <<'PY'
import json, collections, re, sys

# Re-implement the production logic verbatim so we can probe it.
sys.path.insert(0, "training/physics")
from rejection_sample import extract_answer, normalize_answer, check_correctness

TRACES = "/work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl"
PROBLEMS = "/work/hdd/bgde/jhill5/data/physics_problems.jsonl"

# 1. Pull problem-level metadata so we can correlate by problem_id.
problems = {}
with open(PROBLEMS) as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("id")
        if pid:
            problems[pid] = r

print(f"[problems] loaded {len(problems)} problem records")
print(f"[problems] sample keys: {sorted(next(iter(problems.values())).keys())}")
print()

# 2. Walk traces, classify by answer-field presence and domain.
no_ans_by_domain    = collections.Counter()
has_ans_by_domain   = collections.Counter()
has_ans_correct     = 0
has_ans_wrong       = []   # collect first 20 mismatches
has_ans_no_extract  = 0    # answer field present but extract_answer returned None
sample_no_ans       = []

with open(TRACES) as f:
    for line in f:
        r = json.loads(line)
        ans = r.get("answer", "")
        domain = r.get("domain", "?")
        if not ans:
            no_ans_by_domain[domain] += 1
            if len(sample_no_ans) < 6:
                sample_no_ans.append(r)
        else:
            has_ans_by_domain[domain] += 1
            predicted = extract_answer(r.get("trace", ""))
            if predicted is None:
                has_ans_no_extract += 1
                if len(has_ans_wrong) < 20:
                    has_ans_wrong.append(("NO_EXTRACT", r, None))
            else:
                if normalize_answer(predicted) == normalize_answer(ans):
                    has_ans_correct += 1
                else:
                    if len(has_ans_wrong) < 20:
                        has_ans_wrong.append(("MISMATCH", r, predicted))

print("=" * 70)
print(f"BUG A — `answer` field missing/empty:")
print(f"  total NO ANSWER : {sum(no_ans_by_domain.values()):5d}")
print(f"  total HAS ANSWER: {sum(has_ans_by_domain.values()):5d}")
print()
print("  NO_ANSWER by domain (top 12):")
for k, v in no_ans_by_domain.most_common(12):
    print(f"    {v:5d}  {k}")
print()
print("  HAS_ANSWER by domain (top 12):")
for k, v in has_ans_by_domain.most_common(12):
    print(f"    {v:5d}  {k}")

print()
print("=" * 70)
print("Sample 'no answer' traces — does the corresponding PROBLEM record")
print("have ground truth that was just dropped during trace generation?")
print()
for r in sample_no_ans[:6]:
    pid = r.get("problem_id", "?")
    p = problems.get(pid, {})
    print(f"  trace.problem_id = {pid!r}")
    print(f"    trace.domain        = {r.get('domain')!r}")
    print(f"    trace.answer (raw)  = {r.get('answer')!r}")
    print(f"    problem.id          = {p.get('id')!r}")
    print(f"    problem keys        = {sorted(p.keys())}")
    # Look for any field that smells like ground truth on the problem record
    for key in ("answer", "correct", "correct_answer", "label", "choice_a",
                "choice_b", "choice_c", "choice_d", "solution"):
        if key in p:
            v = p[key]
            short = (v[:80] + "…") if isinstance(v, str) and len(v) > 80 else v
            print(f"    problem.{key:18s} = {short!r}")
    print()

print("=" * 70)
print(f"BUG B — `answer` present, but correctness check fails:")
print(f"  HAS_ANSWER total : {sum(has_ans_by_domain.values()):5d}")
print(f"    correct         : {has_ans_correct:5d}")
print(f"    extract failed  : {has_ans_no_extract:5d}")
print(f"    extract!=ground : {len(has_ans_wrong) - has_ans_no_extract:5d} (capped at 20 for display)")
print()
print("  First 12 mismatches — show GROUND TRUTH vs PREDICTED vs trace tail:")
print()
for tag, r, predicted in has_ans_wrong[:12]:
    gt = r.get("answer", "")
    trace_text = r.get("trace", "")
    tail = trace_text[-300:].replace("\n", "\\n")
    print(f"  [{tag}] problem_id={r.get('problem_id')!r}  domain={r.get('domain')!r}")
    print(f"    ground truth (raw)        : {gt!r}")
    print(f"    ground truth (normalized) : {normalize_answer(gt)!r}")
    print(f"    extracted (raw)           : {predicted!r}")
    if predicted is not None:
        print(f"    extracted (normalized)    : {normalize_answer(predicted)!r}")
    print(f"    trace tail (last 300 chars): {tail!r}")
    print()
PY
