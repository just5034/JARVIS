#!/usr/bin/env bash
# Run ARIA evaluation against Qwen3.5-27B served via vLLM on Delta.
#
# Starts a vLLM OpenAI-compatible server, waits for it to be healthy,
# then runs ARIA prototype against it.
#
# Budget: ~16 SU (4 GPUs × 4 hours)
#
# v2 changes:
# - Qwen3.5-aware: strip_thinking + extract_boxed_answer from eval base
# - Differentiated token budgets: solve=32K, verify=4K
# - Combined verify+extract into single LLM call
# - Correct sampling: temp=0.6, top_p=0.95 (published defaults)
# - Time limit increased: 6 problems × ~20min each × 2 methods = ~4hr
#
# Usage:
#   sbatch scripts/run_aria_eval.sh

#SBATCH --job-name=jarvis-aria
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/aria-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/aria-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde/jhill5/jarvis-venv"
source "$VENV/bin/activate"

export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp

# ─── Paths ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
EVAL_OUT="/scratch/bgde/jhill5/eval"
PORT=8192
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$EVAL_OUT" /scratch/bgde/jhill5/logs

# ─── Validate ───
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Model not found at $BASE_MODEL"
    exit 1
fi

echo "=== ARIA Evaluation — Qwen3.5-27B ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $(nvidia-smi -L | wc -l)"
echo "Model:  $BASE_MODEL"
echo "Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Start vLLM server in background ───
echo "Starting vLLM server on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port $PORT \
    --disable-log-stats \
    2>&1 | tee /scratch/bgde/jhill5/logs/aria-vllm-${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ─── Wait for vLLM to be ready ───
echo "Waiting for vLLM to start (this takes ~10 minutes for Qwen3.5)..."
MAX_WAIT=1800  # 30 minutes — Qwen3.5 needs ~15-20 min for compile+warmup
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM is ready! (waited ${WAITED}s)"
        break
    fi
    # Check if vLLM crashed
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ─── Run ARIA ───
echo ""
echo "=== Running ARIA v2 (3 passes, solve=32K, verify=4K) ==="
echo "Sampling: temp=0.6, top_p=0.95 (Qwen3.5 published defaults)"
echo ""
python scripts/aria_prototype.py \
    --backend openai \
    --base-url "http://localhost:${PORT}/v1" \
    --model "$BASE_MODEL" \
    --api-key "not-needed" \
    --max-passes 3 \
    --solve-max-tokens 32768 \
    --verify-max-tokens 4096 \
    --output "$EVAL_OUT/aria_qwen35_${TIMESTAMP}.json"

echo ""
echo "=== ARIA evaluation complete ==="
echo "Results: $EVAL_OUT/aria_qwen35_${TIMESTAMP}.json"

# ─── Cleanup ───
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "Done."
