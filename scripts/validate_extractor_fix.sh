#!/usr/bin/env bash
# Validate the new extract_answer regex against cached physics traces.
# Confirms the GPQA correctness rate jumps from 22/1552 (1.4%) to something
# in the 70-90% range BEFORE we burn time re-running the full filter.
#
# Read-only. Login-node safe.

set -uo pipefail

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
TRACES="/work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl"

module load python/3.13.5-gcc13.3.1 >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$HOME/JARVIS"

python <<'PY'
import json, sys
sys.path.insert(0, "training/physics")
from rejection_sample import extract_answer, normalize_answer

TRACES = "/work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl"

correct = wrong = no_extract = no_gt = 0
sample_wrong = []
sample_correct = []

with open(TRACES) as f:
    for line in f:
        r = json.loads(line)
        gt = r.get("answer", "")
        if not gt:
            no_gt += 1
            continue
        pred = extract_answer(r.get("trace", ""))
        if pred is None:
            no_extract += 1
            continue
        if normalize_answer(pred) == normalize_answer(gt):
            correct += 1
            if len(sample_correct) < 5:
                sample_correct.append((gt, pred, r.get("domain")))
        else:
            wrong += 1
            if len(sample_wrong) < 8:
                sample_wrong.append((gt, pred, r.get("domain"),
                                     r.get("trace", "")[-200:].replace("\n", " ")))

print("=" * 70)
print("NEW EXTRACTOR — pass rate against cached traces")
print("=" * 70)
gt_total = correct + wrong + no_extract
print(f"  no ground truth     : {no_gt:5d}  (unchanged)")
print(f"  with ground truth   : {gt_total:5d}")
print(f"    correct           : {correct:5d}  ({100*correct/gt_total:.1f}%)")
print(f"    wrong (mismatch)  : {wrong:5d}  ({100*wrong/gt_total:.1f}%)")
print(f"    no extract        : {no_extract:5d}  ({100*no_extract/gt_total:.1f}%)")
print()
print(f"  baseline (old extractor): 22 correct ({100*22/gt_total:.1f}%)")
print(f"  delta                   : +{correct - 22}")
print()

print("─" * 70)
print("Sample of CORRECTLY matched (sanity-check the patterns are firing):")
for gt, pred, dom in sample_correct:
    print(f"  GT={gt!r:5s}  PRED={pred!r:5s}  domain={dom!r}")
print()
print("─" * 70)
print("Sample of REMAINING mismatches (post-fix — show what's still failing):")
for gt, pred, dom, tail in sample_wrong:
    print(f"  GT={gt!r:5s}  PRED={pred!r:8s}  domain={dom!r}")
    print(f"    tail: {tail!r}")
    print()
PY
