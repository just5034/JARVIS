#!/usr/bin/env bash
# Generate a training progress snapshot that you can share with Claude.
#
# Collects:
#   - Active/recent SLURM jobs
#   - Latest training metrics from logs
#   - Latest eval results
#   - GPU utilization
#   - Disk/SU usage
#
# Usage:
#   # On Delta — print to terminal (copy-paste into Claude chat):
#   bash scripts/training_status.sh
#
#   # Save to file and sync to local repo:
#   bash scripts/training_status.sh > /u/jhill5/jarvis/training_status.txt
#   # Then on local: scp jhill5@login.delta.ncsa.illinois.edu:/u/jhill5/jarvis/training_status.txt .
#
#   # Or save directly into the repo for Claude to read:
#   bash scripts/training_status.sh --save

set -euo pipefail

SCRATCH="/work/hdd/bgde/jhill5"
LOGS="$SCRATCH/logs"
EVAL="$SCRATCH/eval"
CHECKPOINTS="$SCRATCH/checkpoints"
TB_LOGS="$SCRATCH/tb_logs"
REPO_DIR="/u/jhill5/jarvis"

SAVE_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--save" ]; then
        SAVE_MODE=true
    fi
done

# Redirect output to file if --save
if $SAVE_MODE; then
    OUTPUT_FILE="$REPO_DIR/training_status.txt"
    exec > "$OUTPUT_FILE"
    echo "(saved to $OUTPUT_FILE)" >&2
fi

echo "================================================================"
echo "  JARVIS Training Status — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  User: jhill5 | Account: bgde-delta-gpu"
echo "================================================================"
echo ""

# ─── SLURM Jobs ───
echo "=== Active / Recent SLURM Jobs ==="
squeue -u jhill5 --format="%.10i %.20j %.8T %.10M %.10l %.6D %.4C %.20R" 2>/dev/null || echo "  (not on login node)"
echo ""
echo "--- Last 5 completed jobs ---"
sacct -u jhill5 --format="JobID,JobName%20,State,Elapsed,MaxRSS,MaxVMSize" \
    --starttime=$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d 2>/dev/null || echo "2026-03-17") \
    -n 2>/dev/null | head -10 || echo "  (sacct not available)"
echo ""

# ─── SU Balance ───
echo "=== SU Balance ==="
accounts 2>/dev/null | grep -A2 bgde-delta-gpu || echo "  (run 'accounts' manually to check)"
echo ""

# ─── Latest Training Logs ───
echo "=== Latest Training Logs ==="
if [ -d "$LOGS" ]; then
    LATEST_LOG=$(ls -t "$LOGS"/*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "File: $LATEST_LOG"
        echo "Last modified: $(stat -c %y "$LATEST_LOG" 2>/dev/null || stat -f %Sm "$LATEST_LOG" 2>/dev/null)"
        echo ""
        echo "--- Last 30 lines ---"
        tail -30 "$LATEST_LOG"
    else
        echo "  No log files found in $LOGS"
    fi
else
    echo "  Logs directory not found: $LOGS"
fi
echo ""

# ─── Training Metrics (grep for loss/accuracy from latest log) ───
echo "=== Training Metrics (from latest log) ==="
if [ -n "${LATEST_LOG:-}" ] && [ -f "$LATEST_LOG" ]; then
    # Extract loss values — look for common HuggingFace Trainer output patterns
    echo "--- Loss progression ---"
    grep -oP "'loss':\s*[\d.]+" "$LATEST_LOG" 2>/dev/null | tail -10 || \
    grep -oP "loss['\"]?:\s*[\d.]+" "$LATEST_LOG" 2>/dev/null | tail -10 || \
    echo "  (no loss values found in log)"
    echo ""
    echo "--- Learning rate ---"
    grep -oP "'learning_rate':\s*[\d.e-]+" "$LATEST_LOG" 2>/dev/null | tail -3 || \
    echo "  (no LR values found)"
    echo ""
    echo "--- Epoch/Step ---"
    grep -oP "'epoch':\s*[\d.]+" "$LATEST_LOG" 2>/dev/null | tail -3 || \
    echo "  (no epoch info found)"
fi
echo ""

# ─── Eval Results ───
echo "=== Latest Eval Results ==="
if [ -d "$EVAL" ]; then
    for f in "$EVAL"/*.json; do
        if [ -f "$f" ]; then
            echo "--- $(basename "$f") ---"
            # Extract just the metrics block
            python3 -c "
import json, sys
with open('$f') as fh:
    data = json.load(fh)
metrics = data.get('metrics', data)
for k, v in metrics.items():
    print(f'  {k}: {v}')
" 2>/dev/null || echo "  (could not parse)"
            echo ""
        fi
    done
else
    echo "  No eval results yet in $EVAL"
fi
echo ""

# ─── Checkpoints ───
echo "=== Checkpoints ==="
if [ -d "$CHECKPOINTS" ]; then
    for dir in "$CHECKPOINTS"/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            count=$(find "$dir" -maxdepth 1 -type d | wc -l)
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            latest=$(ls -td "$dir"*/ 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "?")
            echo "  $name: $count checkpoints, $size, latest: $latest"
        fi
    done
else
    echo "  No checkpoints yet"
fi
echo ""

# ─── Disk Usage ───
echo "=== Disk Usage ==="
echo "  scratch: $(du -sh "$SCRATCH" 2>/dev/null | cut -f1 || echo '?')"
echo "  home:    $(du -sh /u/jhill5 2>/dev/null | cut -f1 || echo '?')"
echo ""

# ─── GPU Status (if on compute node) ───
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "  (not on a GPU node — run from within a job)"
echo ""

echo "================================================================"
echo "  Copy everything above and paste into Claude chat for analysis."
echo "================================================================"
