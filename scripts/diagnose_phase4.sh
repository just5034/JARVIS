#!/usr/bin/env bash
# Phase 4 post-mortem: figure out what each trace-gen job actually produced,
# whether physics clobbered code, and whether rejection_sample ran.
#
# Read-only. Safe to run on a login node. No SLURM, no SUs.
#
# Usage:
#   bash scripts/diagnose_phase4.sh

set -uo pipefail   # NOT set -e — we want to see every section even if one fails

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
DATA_DIR="/work/hdd/bgde/jhill5/data"
LOG_DIR="/work/hdd/bgde/jhill5/logs"
TRACES_DIR="$DATA_DIR/hep_traces"

PHYSICS_JOB=18023751
CODE_JOB=18023754

hr()  { printf '%s\n' "════════════════════════════════════════════════════════════"; }
sub() { printf '%s\n' "──────────────────────────────────────────"; }
say() { printf '\n[diag] %s\n' "$*"; }

# ─── env ───
module load python/3.13.5-gcc13.3.1 >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

hr
echo "PHASE 4 POST-MORTEM"
hr

# ─── 1. SLURM accounting ───
say "1. SLURM accounting for both jobs"
sacct -j $PHYSICS_JOB,$CODE_JOB \
    --format=JobID,JobName%20,State,ExitCode,Elapsed,Submit,End \
    -P 2>/dev/null | column -t -s '|'

# ─── 2. What's in hep_traces/ ───
say "2. /work/hdd/bgde/jhill5/data/hep_traces — file inventory"
ls -la "$TRACES_DIR" 2>/dev/null

# ─── 3. Other recent JSONLs anywhere under data/ ───
say "3. Recently-modified JSONLs under $DATA_DIR (since Apr 30, excludes benchmarks/)"
find "$DATA_DIR" -name '*.jsonl' -newermt '2026-04-30' \
    ! -path '*/benchmarks/*' \
    -exec ls -la {} \; 2>/dev/null

# ─── 4. Source / domain breakdown of traces.jsonl ───
say "4. Source + domain distribution in traces.jsonl"
python <<'PY' 2>/dev/null
import json, collections, os
path = "/work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl"
if not os.path.exists(path):
    print(f"  MISSING: {path}")
    raise SystemExit
src = collections.Counter()
dom = collections.Counter()
total = bad = 0
sample_keys = None
with open(path) as f:
    for line in f:
        total += 1
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        if sample_keys is None:
            sample_keys = sorted(r.keys())
        src[r.get("source", "?")] += 1
        dom[r.get("domain", "?")] += 1
print(f"  total lines: {total}  ({bad} unparseable)")
print(f"  record keys: {sample_keys}")
print()
print("  SOURCES (top 15):")
for k, v in src.most_common(15):
    print(f"    {v:6d}  {k}")
print()
print("  DOMAINS:")
for k, v in dom.most_common():
    print(f"    {v:6d}  {k}")
PY

# ─── 5. Did either job's stdout mention rejection sampling / filtering? ───
say "5. Did rejection_sample.py run? (grep .out files)"
for job_id in $PHYSICS_JOB $CODE_JOB; do
    sub
    # Both naming conventions exist (script comments differ). Match either.
    out_file=$(ls -t "$LOG_DIR"/hep-*-traces-${job_id}.out 2>/dev/null | head -1)
    [ -z "$out_file" ] && out_file=$(ls -t "$LOG_DIR"/hep-traces-${job_id}.out 2>/dev/null | head -1)
    if [ -z "$out_file" ]; then
        echo "  job $job_id: no .out file found"
        continue
    fi
    echo "  job $job_id  -->  $out_file"
    echo "  --- last 30 lines ---"
    tail -30 "$out_file"
    echo "  --- grep for filter/kept/rejected/wrote/target ---"
    grep -niE 'filter|kept|rejected|wrote|target-count|--domain|--output' "$out_file" | tail -20
done

# ─── 6. Where the trace-gen scripts wrote things ───
say "6. What did the scripts CONFIGURE as output paths?"
sub
echo "  scripts/run_trace_generation.sh:"
grep -E 'OUTPUT_DIR=|--output|rejection_sample|target-count' \
    scripts/run_trace_generation.sh | sed 's/^/    /'
sub
echo "  scripts/run_code_trace_generation.sh:"
grep -E 'OUTPUT_DIR=|--output|rejection_sample|target-count' \
    scripts/run_code_trace_generation.sh | sed 's/^/    /'

# ─── 7. Tail stderr too (per feedback_check_stderr.md) ───
say "7. Stderr tails (last 20 non-matplotlib lines)"
for job_id in $PHYSICS_JOB $CODE_JOB; do
    sub
    err_file=$(ls -t "$LOG_DIR"/hep-*-traces-${job_id}.err 2>/dev/null | head -1)
    [ -z "$err_file" ] && err_file=$(ls -t "$LOG_DIR"/hep-traces-${job_id}.err 2>/dev/null | head -1)
    if [ -z "$err_file" ]; then
        echo "  job $job_id: no .err file"
        continue
    fi
    echo "  job $job_id  -->  $err_file"
    grep -v 'matplotlib-3.8.0' "$err_file" \
        | grep -vE '^(  Traceback|    File|  AttributeError|Remainder|$)' \
        | tail -20
done

hr
echo "DIAGNOSIS DONE"
hr
