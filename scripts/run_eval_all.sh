#!/usr/bin/env bash
# Run ALL benchmark evaluations on Delta.
# Evaluates physics, math, code brains and router, logging to TensorBoard.
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
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde-delta-gpu/logs/eval-%j.out
#SBATCH --error=/scratch/bgde-delta-gpu/logs/eval-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/u/$USER/jarvis-venv"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -e "/u/$USER/jarvis[serving,training]"
else
    source "$VENV/bin/activate"
fi

export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp

# ─── Paths ───
MODELS="/projects/bgde-delta-gpu/models"
ADAPTERS="/projects/bgde-delta-gpu/adapters"
DATA="/scratch/bgde-delta-gpu/data/benchmarks"
EVAL_OUT="/scratch/bgde-delta-gpu/eval"
TB_LOGS="/scratch/bgde-delta-gpu/tb_logs"

mkdir -p "$EVAL_OUT" "$TB_LOGS"

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

echo "=== JARVIS Evaluation Suite ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $(nvidia-smi -L | wc -l)"
echo "Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── GPQA Diamond (Physics Brain) ───
if $RUN_PHYSICS; then
    echo "=== [1/4] GPQA Diamond — Physics Brain ==="
    PHYSICS_MODEL="${MODELS}/r1-distill-qwen-32b"
    PHYSICS_ADAPTER="${ADAPTERS}/physics_general"

    ADAPTER_FLAG=""
    if [ -d "$PHYSICS_ADAPTER" ]; then
        ADAPTER_FLAG="--adapter $PHYSICS_ADAPTER"
    fi

    python -m training.eval.run_gpqa \
        --model "$PHYSICS_MODEL" \
        $ADAPTER_FLAG \
        --output "$EVAL_OUT/gpqa_diamond_$(date +%Y%m%d).json" \
        --data-dir "$DATA" \
        --log-dir "$TB_LOGS" \
        --experiment "physics_eval"
    echo ""
fi

# ─── AIME 2024 (Math Brain) ───
if $RUN_MATH; then
    echo "=== [2/4] AIME 2024 — Math Brain ==="
    MATH_MODEL="${MODELS}/r1-distill-qwen-32b"

    python -m training.eval.run_aime \
        --model "$MATH_MODEL" \
        --output "$EVAL_OUT/aime_2024_$(date +%Y%m%d).json" \
        --data-dir "$DATA" \
        --log-dir "$TB_LOGS" \
        --experiment "math_eval" \
        --max-tokens 8192
    echo ""
fi

# ─── LiveCodeBench (Code Brain) ───
if $RUN_CODE; then
    echo "=== [3/4] LiveCodeBench — Code Brain ==="
    CODE_MODEL="${MODELS}/qwen3-32b"
    CODE_ADAPTER="${ADAPTERS}/code_general"

    ADAPTER_FLAG=""
    if [ -d "$CODE_ADAPTER" ]; then
        ADAPTER_FLAG="--adapter $CODE_ADAPTER"
    fi

    python -m training.eval.run_livecode \
        --model "$CODE_MODEL" \
        $ADAPTER_FLAG \
        --output "$EVAL_OUT/livecode_$(date +%Y%m%d).json" \
        --data-dir "$DATA" \
        --log-dir "$TB_LOGS" \
        --experiment "code_eval"
    echo ""
fi

# ─── Router Classifier ───
if $RUN_ROUTER; then
    echo "=== [4/4] Router Classifier ==="
    ROUTER_MODEL="${MODELS}/router_bert"

    if [ -d "$ROUTER_MODEL" ]; then
        python -m training.eval.run_router_eval \
            --router-model "$ROUTER_MODEL" \
            --data-dir "$DATA" \
            --output "$EVAL_OUT/router_$(date +%Y%m%d).json" \
            --log-dir "$TB_LOGS"
    else
        echo "  SKIP: router model not found at $ROUTER_MODEL"
    fi
    echo ""
fi

echo "=== Evaluation Complete ==="
echo "Results: $EVAL_OUT"
echo ""
echo "To view TensorBoard from your local machine:"
echo "  ssh -L 6006:localhost:6006 jhill5@login.delta.ncsa.illinois.edu"
echo "  # Then run on Delta: tensorboard --logdir $TB_LOGS --port 6006"
echo "  # Open: http://localhost:6006"
