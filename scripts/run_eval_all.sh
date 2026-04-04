#!/usr/bin/env bash
# Run ALL benchmark evaluations on Delta with Qwen3.5-27B.
# Evaluates single unified base model on physics, math, and code benchmarks.
#
# Budget: ~32 SU (4 GPUs × 8 hours)
#
# Usage:
#   sbatch scripts/run_eval_all.sh
#   # Or run specific benchmarks:
#   sbatch scripts/run_eval_all.sh --physics-only
#   sbatch scripts/run_eval_all.sh --code-only

#SBATCH --job-name=jarvis-eval
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/eval-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/eval-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde/jhill5/jarvis-venv"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -e "/u/$USER/JARVIS[serving,training]"
else
    source "$VENV/bin/activate"
fi

export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp

# ─── Paths ───
# Single unified base model — all benchmarks use the same model
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
DATA="/scratch/bgde/jhill5/data/benchmarks"
EVAL_OUT="/scratch/bgde/jhill5/eval"
TB_LOGS="/scratch/bgde/jhill5/tb_logs"
ROUTER_MODEL="/projects/bgde/jhill5/models/router_bert"

mkdir -p "$EVAL_OUT" "$TB_LOGS"

# ─── Validate model ───
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Qwen3.5-27B not found at $BASE_MODEL"
    echo "Run: bash scripts/download_qwen35.sh"
    exit 1
fi

# ─── Download benchmarks if needed ───
if [ ! -f "$DATA/manifest.json" ]; then
    echo "=== Downloading benchmark data ==="
    python -m training.data.download_benchmarks --output "$DATA"
fi

# ─── Parse args ───
RUN_PHYSICS=true
RUN_MATH=true
RUN_CODE=true
RUN_ROUTER=true

for arg in "$@"; do
    case $arg in
        --physics-only) RUN_MATH=false; RUN_CODE=false; RUN_ROUTER=false ;;
        --math-only)    RUN_PHYSICS=false; RUN_CODE=false; RUN_ROUTER=false ;;
        --code-only)    RUN_PHYSICS=false; RUN_MATH=false; RUN_ROUTER=false ;;
        --router-only)  RUN_PHYSICS=false; RUN_MATH=false; RUN_CODE=false ;;
    esac
done

echo "=== JARVIS Baseline Evaluation — Qwen3.5-27B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $(nvidia-smi -L | wc -l)"
echo "Model:  $BASE_MODEL"
echo "Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Published targets:"
echo "  GPQA Diamond:  86%"
echo "  AIME 2024:     81%"
echo "  LiveCodeBench:  80.7%"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ─── GPQA Diamond (Physics) ───
if $RUN_PHYSICS; then
    echo "=== [1/4] GPQA Diamond — Physics ==="
    echo "  Target: >= 86% (published Qwen3.5-27B)"
    python -m training.eval.run_gpqa \
        --model "$BASE_MODEL" \
        --output "$EVAL_OUT/gpqa_diamond_qwen35_${TIMESTAMP}.json" \
        --data-dir "$DATA" \
        --log-dir "$TB_LOGS" \
        --experiment "qwen35_baseline_gpqa" \
        --temperature 0.6

    # Print result summary
    python -c "
import json
with open('$EVAL_OUT/gpqa_diamond_qwen35_${TIMESTAMP}.json') as f:
    r = json.load(f)
m = r.get('metrics', {})
acc = m.get('accuracy', 0) * 100
n = m.get('n_total', 0)
print(f'  Result: {acc:.1f}% ({m.get(\"n_correct\", 0)}/{n})')
target = 86.0
if acc >= target * 0.95:  # within 5% of target
    print(f'  STATUS: PASS (>= {target*0.95:.1f}% threshold)')
else:
    print(f'  STATUS: BELOW TARGET (expected >= {target*0.95:.1f}%)')
" 2>/dev/null || echo "  (could not parse results)"
    echo ""
fi

# ─── AIME 2024 (Math) ───
if $RUN_MATH; then
    echo "=== [2/4] AIME 2024 — Math ==="
    echo "  Target: >= 81% (published Qwen3.5-27B)"
    python -m training.eval.run_aime \
        --model "$BASE_MODEL" \
        --output "$EVAL_OUT/aime_2024_qwen35_${TIMESTAMP}.json" \
        --data-dir "$DATA" \
        --log-dir "$TB_LOGS" \
        --experiment "qwen35_baseline_aime" \
        --temperature 0.0

    python -c "
import json
with open('$EVAL_OUT/aime_2024_qwen35_${TIMESTAMP}.json') as f:
    r = json.load(f)
m = r.get('metrics', {})
acc = m.get('accuracy', 0) * 100
n = m.get('n_total', 0)
print(f'  Result: {acc:.1f}% ({m.get(\"n_correct\", 0)}/{n})')
target = 81.0
if acc >= target * 0.90:
    print(f'  STATUS: PASS (>= {target*0.90:.1f}% threshold)')
else:
    print(f'  STATUS: BELOW TARGET (expected >= {target*0.90:.1f}%)')
" 2>/dev/null || echo "  (could not parse results)"
    echo ""
fi

# ─── LiveCodeBench (Code) ───
if $RUN_CODE; then
    echo "=== [3/4] LiveCodeBench — Code ==="
    echo "  Target: >= 80.7% (published Qwen3.5-27B)"
    python -m training.eval.run_livecode \
        --model "$BASE_MODEL" \
        --output "$EVAL_OUT/livecode_qwen35_${TIMESTAMP}.json" \
        --data-dir "$DATA" \
        --log-dir "$TB_LOGS" \
        --experiment "qwen35_baseline_livecode" \
        --max-tokens 16384

    python -c "
import json
with open('$EVAL_OUT/livecode_qwen35_${TIMESTAMP}.json') as f:
    r = json.load(f)
m = r.get('metrics', {})
acc = m.get('pass_at_1', m.get('accuracy', 0)) * 100
n = m.get('n_total', 0)
print(f'  Result: {acc:.1f}% pass@1 ({n} problems)')
target = 80.7
if acc >= target * 0.90:
    print(f'  STATUS: PASS (>= {target*0.90:.1f}% threshold)')
else:
    print(f'  STATUS: BELOW TARGET (expected >= {target*0.90:.1f}%)')
" 2>/dev/null || echo "  (could not parse results)"
    echo ""
fi

# ─── Router Classifier ───
if $RUN_ROUTER; then
    echo "=== [4/4] Router Classifier ==="
    if [ -d "$ROUTER_MODEL" ]; then
        python -m training.eval.run_router_eval \
            --router-model "$ROUTER_MODEL" \
            --data-dir "$DATA" \
            --output "$EVAL_OUT/router_qwen35_${TIMESTAMP}.json" \
            --log-dir "$TB_LOGS"
    else
        echo "  SKIP: router model not found at $ROUTER_MODEL (expected — using heuristic router)"
    fi
    echo ""
fi

# ─── Summary ───
echo "=========================================="
echo "  BASELINE EVALUATION COMPLETE"
echo "=========================================="
echo "Results directory: $EVAL_OUT"
echo ""
echo "Result files:"
ls -la "$EVAL_OUT"/*qwen35*${TIMESTAMP}* 2>/dev/null || echo "  (no result files found)"
echo ""
echo "To view TensorBoard from your local machine:"
echo "  ssh -L 6006:localhost:6006 jhill5@login.delta.ncsa.illinois.edu"
echo "  # Then on Delta: tensorboard --logdir $TB_LOGS --port 6006"
echo "  # Open: http://localhost:6006"
echo ""
echo "Next steps:"
echo "  - If baselines match published numbers: proceed to HEP LoRA training"
echo "  - If below target: check vLLM settings, quantization, temperature"
