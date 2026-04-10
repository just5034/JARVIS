#!/usr/bin/env bash
# Tool-use experiment: vLLM OpenAI server with structured tool-call support.
#
# Distinct from run_vllm_compat.sh — different job name, different port,
# different log files. Safe to run alongside the migration eval jobs.
#
# Budget: ~4 SU (4 GPUs × 1 hour) for the smoke test.
#
# Usage:
#   sbatch scripts/run_vllm_tooluse.sh

#SBATCH --job-name=jarvis-tooluse
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/tooluse-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/tooluse-%j.err

set -euo pipefail

# ─── Environment (matches run_vllm_compat.sh) ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde/jhill5/jarvis-venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: venv $VENV does not exist. Bootstrap with run_vllm_compat.sh first."
    exit 1
fi
source "$VENV/bin/activate"

# Ensure the tooluse package from this branch is installed, plus pytest for smoke tests
pip install -e "/u/$USER/JARVIS" pytest httpx --quiet 2>&1 | tail -3

mkdir -p /scratch/bgde/jhill5/logs

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# ─── Paths and ports (distinct from compat job) ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
VLLM_PORT=8290
PROXY_PORT=8001
VLLM_LOG="/scratch/bgde/jhill5/logs/tooluse-vllm-${SLURM_JOB_ID}.log"
PROXY_LOG="/scratch/bgde/jhill5/logs/tooluse-proxy-${SLURM_JOB_ID}.log"

echo "=== Tool-use experiment ==="
echo "vLLM:    $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo NOT_INSTALLED)"
echo "Model:   $BASE_MODEL"
echo "vLLM port:  $VLLM_PORT"
echo "Proxy port: $PROXY_PORT"
echo

# ─── Probe vLLM for the correct tool-call parser name ───
# Different vLLM versions ship different parser identifiers for Qwen3-family
# models (commonly: hermes, qwen, or qwen3). Pick the first one that --help
# advertises so the launch doesn't fail on a stale name.
PARSER_OPTS=$(python -m vllm.entrypoints.openai.api_server --help 2>&1 | \
              grep -A 20 "tool-call-parser" | tr -d ',' || true)
echo "Available parsers (raw):"
echo "$PARSER_OPTS" | head -5
TOOL_PARSER=""
for candidate in qwen3_xml qwen3_coder hermes; do
    if echo "$PARSER_OPTS" | grep -qw "$candidate"; then
        TOOL_PARSER="$candidate"
        break
    fi
done
if [ -z "$TOOL_PARSER" ]; then
    echo "WARNING: could not auto-detect tool parser; defaulting to 'qwen3_xml'"
    TOOL_PARSER="qwen3_xml"
fi
echo "Using --tool-call-parser=$TOOL_PARSER"
echo

# ─── Launch vLLM OpenAI server with tool-call support ───
echo "=== Launching vLLM (log: $VLLM_LOG) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM (PID $VLLM_PID)..."
for i in $(seq 1 1800); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "  vLLM up after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  vLLM died during startup. Last 40 lines:"
        tail -40 "$VLLM_LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "vLLM did not become ready within 1800s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ─── Launch the proxy ───
echo "=== Launching tool-use proxy on :$PROXY_PORT (log: $PROXY_LOG) ==="
JARVIS_TOOLUSE_VLLM_URL="http://localhost:$VLLM_PORT" \
    python -m jarvis.tooluse.server --host 0.0.0.0 --port $PROXY_PORT \
    > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!

sleep 5
if ! curl -s "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
    echo "Proxy failed to start. Last 40 lines:"
    tail -40 "$PROXY_LOG"
    kill $VLLM_PID $PROXY_PID 2>/dev/null || true
    exit 1
fi
echo "Proxy healthy."

# ─── Smoke test ───
# Disable set -e so we can capture the exit code and still run teardown.
echo "=== Running smoke test ==="
set +e
JARVIS_TOOLUSE_PROXY_URL="http://localhost:$PROXY_PORT" \
    python -m pytest tests/tooluse/test_smoke.py -v -s
SMOKE_RC=$?
set -e

# ─── Teardown ───
echo "=== Shutting down ==="
kill $PROXY_PID 2>/dev/null || true
kill $VLLM_PID 2>/dev/null || true
wait 2>/dev/null || true

echo
echo "Smoke test exit code: $SMOKE_RC"
echo "vLLM log:  $VLLM_LOG"
echo "Proxy log: $PROXY_LOG"
exit $SMOKE_RC
