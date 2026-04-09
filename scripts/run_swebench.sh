#!/usr/bin/env bash
# Run SWE-bench Verified prediction generation against Qwen3.5-27B on Delta.
#
# First run: N=20 instances to validate the pipeline (~3-5 hours)
# Full run:  remove --n-instances flag (~80-100 hours, split into chunks)
#
# Output: predictions JSON. Scoring is a SEPARATE step requiring Docker.
#
# Usage:
#   sbatch scripts/run_swebench.sh                # default N=20 baseline
#   sbatch scripts/run_swebench.sh --n 50         # custom N
#   sbatch scripts/run_swebench.sh --resume       # resume from existing predictions

#SBATCH --job-name=jarvis-swebench
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/swebench-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/swebench-%j.err

set -euo pipefail

# ─── Args ───
N_INSTANCES=20
RESUME=false
for arg in "$@"; do
    case $arg in
        --n=*) N_INSTANCES="${arg#*=}" ;;
        --n)   shift; N_INSTANCES="$1" ;;
        --resume) RESUME=true ;;
    esac
done

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde/jhill5/jarvis-venv"
source "$VENV/bin/activate"

# Make sure openai client is installed (used by the agent)
pip install --quiet openai datasets 2>&1 | tail -5 || true

export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp

# ─── Paths ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
EVAL_OUT="/scratch/bgde/jhill5/eval"
WORKDIR="/scratch/bgde/jhill5/swebench_workspaces"
PORT=8193  # different from ARIA's 8192 to avoid clashes
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$EVAL_OUT" "$WORKDIR" /scratch/bgde/jhill5/logs

# Determine output path (resume uses the latest existing file)
if $RESUME; then
    LATEST=$(ls -t "$EVAL_OUT"/swebench_qwen35_*.json 2>/dev/null | head -1 || true)
    if [ -n "$LATEST" ]; then
        OUTPUT="$LATEST"
        echo "Resuming from $OUTPUT"
    else
        OUTPUT="$EVAL_OUT/swebench_qwen35_${TIMESTAMP}.json"
        echo "No existing predictions found, starting fresh"
    fi
else
    OUTPUT="$EVAL_OUT/swebench_qwen35_${TIMESTAMP}.json"
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Model not found at $BASE_MODEL"
    exit 1
fi

echo "=== SWE-bench Verified Eval — Qwen3.5-27B ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L | wc -l)"
echo "Model:     $BASE_MODEL"
echo "N instances: $N_INSTANCES"
echo "Output:    $OUTPUT"
echo "Workdir:   $WORKDIR"
echo "Date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Start vLLM server ───
echo "Starting vLLM OpenAI-compatible server on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port $PORT \
    --disable-log-stats \
    > /scratch/bgde/jhill5/logs/swebench-vllm-${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ─── Wait for vLLM ───
echo "Waiting for vLLM (~15-20 min for Qwen3.5)..."
MAX_WAIT=1800
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died — check /scratch/bgde/jhill5/logs/swebench-vllm-${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ─── Run prediction generation ───
echo ""
echo "=== Running SWE-bench prediction agent ==="
echo "Sampling: temp=0.6, top_p=0.95, max_tokens=4096"
echo ""

EXTRA_ARGS=""
if [ "$N_INSTANCES" != "all" ]; then
    EXTRA_ARGS="--n-instances $N_INSTANCES"
fi

python -m training.eval.run_swebench \
    --base-url "http://localhost:${PORT}/v1" \
    --model "$BASE_MODEL" \
    --api-key "not-needed" \
    --output "$OUTPUT" \
    --workdir "$WORKDIR" \
    --max-steps 25 \
    --max-tokens 4096 \
    --temperature 0.6 \
    $EXTRA_ARGS

# ─── Cleanup ───
echo ""
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "  SWE-BENCH PREDICTIONS COMPLETE"
echo "=========================================="
echo "Predictions: $OUTPUT"
echo ""
echo "Quick stats:"
python -c "
import json
with open('$OUTPUT') as f:
    preds = json.load(f)
total = len(preds)
finished = sum(1 for p in preds if p.get('finished'))
empty = sum(1 for p in preds if not p.get('model_patch','').strip())
errored = sum(1 for p in preds if p.get('error'))
print(f'  Total predictions:  {total}')
print(f'  Finished cleanly:   {finished}')
print(f'  Empty patches:      {empty}')
print(f'  Errored:            {errored}')
" 2>/dev/null || echo "  (could not compute stats)"
echo ""
echo "Next step: score predictions with the official swebench harness:"
echo "  pip install swebench"
echo "  python -m swebench.harness.run_evaluation \\"
echo "      --predictions_path $OUTPUT \\"
echo "      --max_workers 4 --run_id qwen35_baseline \\"
echo "      --dataset_name princeton-nlp/SWE-bench_Verified"
echo "(Requires Docker; can run on a separate machine.)"
