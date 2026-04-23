#!/usr/bin/env bash
# Phase 4A-new: HEP data curation — generate physics reasoning traces
# using Qwen3.5-27B as teacher on A100 GPUs.
#
# Generates 8 traces per problem for HEP-specific training data.
# After generation, runs rejection sampling to filter to best traces.
#
# Budget: ~50 SU (4 GPUs × ~12 hours)
#
# Prerequisites:
#   - Model: /projects/bgde/jhill5/models/qwen3.5-27b
#   - Physics problems: /work/hdd/bgde/jhill5/data/physics_problems.jsonl
#   - Baseline evals passed (run_eval_all.sh)
#
# Usage:
#   sbatch scripts/run_trace_generation.sh

#SBATCH --job-name=jarvis-hep-traces
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/work/hdd/bgde/jhill5/logs/hep-traces-%j.out
#SBATCH --error=/work/hdd/bgde/jhill5/logs/hep-traces-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
source "$VENV/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TMPDIR=/tmp
export HF_HOME=/work/hdd/bgde/jhill5/hf_cache
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export VLLM_LOGGING_LEVEL=WARNING

# ─── Paths ───
TEACHER_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
PROBLEMS="/work/hdd/bgde/jhill5/data/physics_problems.jsonl"
OUTPUT_DIR="/work/hdd/bgde/jhill5/data/hep_traces"
VLLM_PORT=8192

echo "=== JARVIS Phase 4A-new: HEP Trace Generation (Qwen3.5-27B) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L | wc -l)"
echo "Model:   $TEACHER_MODEL"
echo "Problems: $(wc -l < $PROBLEMS 2>/dev/null || echo 'FILE NOT FOUND') problems"
echo "Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Validate inputs ───
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: Qwen3.5-27B not found at $TEACHER_MODEL"
    echo "Run: bash scripts/download_qwen35.sh"
    exit 1
fi

if [ ! -f "$PROBLEMS" ]; then
    echo "ERROR: physics problems not found at $PROBLEMS"
    echo "Generate with: python -m training.data.build_physics_problems"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" /work/hdd/bgde/jhill5/logs

# ─── Sanity checks ───
echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ─── Start vLLM server in background ───
VLLM_LOG="/work/hdd/bgde/jhill5/logs/vllm-hep-traces-${SLURM_JOB_ID}.log"
echo "Starting vLLM server with tensor parallelism=4..."
echo "vLLM log: $VLLM_LOG"
python -m vllm.entrypoints.openai.api_server \
    --model "$TEACHER_MODEL" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --port $VLLM_PORT \
    --no-enable-log-requests \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Wait for server (up to 30 min)
echo "Waiting for vLLM server to start..."
for i in $(seq 1 360); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  vLLM server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died during startup"
        echo "=== Last 50 lines of vLLM log ==="
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start after 1800s"
    tail -100 "$VLLM_LOG"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== Generating HEP traces (8 per problem) ==="
python training/physics/generate_traces_api.py \
    --problems "$PROBLEMS" \
    --output "$OUTPUT_DIR" \
    --model "$TEACHER_MODEL" \
    --api-base "http://localhost:$VLLM_PORT/v1" \
    --api-key "dummy" \
    --traces-per-problem 8 \
    --max-tokens 16384 \
    --temperature 0.7 \
    --workers 8 \
    --resume

echo ""
echo "=== Trace generation complete ==="
echo "Output: $OUTPUT_DIR"

# ─── Shutdown vLLM ───
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# ─── Run rejection sampling ───
echo ""
echo "=== Running rejection sampling ==="
python training/physics/rejection_sample.py \
    --traces "$OUTPUT_DIR/traces.jsonl" \
    --output "/work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl" \
    --target-count 5000 \
    --require-correct

echo ""
echo "=== Phase 4A-new Complete ==="
echo "Raw traces: $OUTPUT_DIR/traces.jsonl"
echo "Filtered traces: /work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl"
echo ""
echo "Next: sbatch scripts/run_hep_sft.sh"
