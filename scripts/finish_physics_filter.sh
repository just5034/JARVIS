#!/usr/bin/env bash
# Run the physics rejection-sampling step that the timed-out SLURM job
# (18023751) didn't reach. CPU-only — safe on a login node, no SUs.
#
# Reads:  /work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl  (6896 raw)
# Writes: /work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl
#
# This mirrors the final block of scripts/run_trace_generation.sh.

set -euo pipefail

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
RAW="/work/hdd/bgde/jhill5/data/hep_traces/traces.jsonl"
FILTERED="/work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl"

echo "=== Phase 4A-new: rejection sampling (post-hoc) ==="
echo "  raw:      $RAW"
echo "  filtered: $FILTERED"

if [ ! -f "$RAW" ]; then
    echo "ERROR: raw traces not found at $RAW"
    exit 1
fi
echo "  raw line count: $(wc -l < "$RAW")"
echo

module load python/3.13.5-gcc13.3.1
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$HOME/JARVIS"

python training/physics/rejection_sample.py \
    --traces "$RAW" \
    --output "$FILTERED" \
    --target-count 5000 \
    --require-correct

echo
echo "=== Done ==="
if [ -f "$FILTERED" ]; then
    echo "  output line count: $(wc -l < "$FILTERED")"
    ls -la "$FILTERED"
fi
echo
echo "Next: sbatch scripts/run_hep_sft.sh --physics"
echo "      sbatch scripts/run_hep_sft.sh --code"
