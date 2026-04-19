#!/usr/bin/env bash
# Run SWE-bench Verified using mini-swe-agent + Qwen3.5-27B on Delta.
#
# Uses the validated mini-swe-agent framework (250 steps, bash-only)
# with Apptainer containers instead of Docker.
#
# Usage:
#   sbatch scripts/run_swebench_mini.sh                    # default: 5 instances (validation)
#   sbatch scripts/run_swebench_mini.sh --slice "0:20"     # first 20 instances
#   sbatch scripts/run_swebench_mini.sh --slice "0:500"    # full run (needs longer --time)

#SBATCH --job-name=jarvis-mswea
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --constraint="projects"
#SBATCH --output=/work/hdd/bgde/jhill5/logs/mswea-%j.out
#SBATCH --error=/work/hdd/bgde/jhill5/logs/mswea-%j.err

set -euo pipefail

# ─── Args ───
# Usage: sbatch scripts/run_swebench_mini.sh          -> 5 instances
#        sbatch scripts/run_swebench_mini.sh 50        -> 50 instances
#        sbatch scripts/run_swebench_mini.sh 0:500     -> full run
N="${1:-0:5}"
SLICE="--slice ${N}"

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8
source /work/hdd/bgde/jhill5/jarvis-venv/bin/activate

export HF_HOME=/work/hdd/bgde/jhill5/hf_cache
export TMPDIR=/tmp  # Apptainer sandboxes build here (node-local SSD)
export MSWEA_COST_TRACKING="ignore_errors"
export MSWEA_SINGULARITY_EXECUTABLE="apptainer"

# ─── Paths ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8193
RESULTS_DIR="/work/hdd/bgde/jhill5/swebench-results"

mkdir -p "$RESULTS_DIR" /work/hdd/bgde/jhill5/logs

echo "=== SWE-bench Verified — mini-swe-agent + Qwen3.5-27B ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L | wc -l)"
echo "Model:     $BASE_MODEL"
echo "Slice:     $SLICE"
echo "Results:   $RESULTS_DIR"
echo "Date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Start vLLM ───
echo "Starting vLLM on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --port $PORT \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    > /work/hdd/bgde/jhill5/logs/mswea-vllm-${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ─── Wait for vLLM ───
echo "Waiting for vLLM (~20-30 min)..."
MAX_WAIT=2700
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (${WAITED}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FATAL: vLLM died during startup"
        tail -40 /work/hdd/bgde/jhill5/logs/mswea-vllm-${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "FATAL: vLLM did not start within ${MAX_WAIT}s"
    tail -40 /work/hdd/bgde/jhill5/logs/mswea-vllm-${SLURM_JOB_ID}.log
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ─── Verify model name ───
echo ""
echo "vLLM model name:"
curl -s "http://localhost:${PORT}/v1/models" | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "(could not fetch)"

# ─── Run mini-swe-agent ───
echo ""
echo "=== Running mini-swe-agent ==="
mini-extra swebench \
    --model "hosted_vllm/${BASE_MODEL}" \
    --subset verified \
    --split test \
    --output "$RESULTS_DIR/run_${SLURM_JOB_ID}" \
    $SLICE \
    --environment-class singularity \
    -c swebench.yaml \
    -c model.model_kwargs.api_base="http://localhost:${PORT}/v1" \
    -c model.model_kwargs.drop_params=true \
    -c model.model_kwargs.temperature=1.0 \
    -c model.cost_tracking=ignore_errors

echo ""
echo "=== mini-swe-agent complete ==="
echo "Results: $RESULTS_DIR/run_${SLURM_JOB_ID}"
ls -la "$RESULTS_DIR/run_${SLURM_JOB_ID}"/preds.json 2>/dev/null || echo "(no preds.json found)"

# ─── Cleanup ───
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
