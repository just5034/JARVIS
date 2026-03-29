#!/usr/bin/env bash
# Phase 4A: Self-distillation — generate physics reasoning traces
# using R1-Distill-Qwen-32B as its own teacher on A100 GPUs.
#
# Generates 8 traces per problem (866 problems × 8 = ~6,928 traces).
# After generation, runs rejection sampling to filter to best traces.
#
# Budget: ~200-350 SU (4 GPUs × 50-90 hrs × 1 SU/GPU-hr)
#
# Prerequisites:
#   - Model: /projects/bgde/jhill5/models/r1-distill-qwen-32b
#   - Physics problems: /scratch/bgde/jhill5/data/physics_problems.jsonl
#
# Usage:
#   sbatch scripts/run_trace_generation.sh

#SBATCH --job-name=jarvis-traces
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/traces-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/traces-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde/jhill5/jarvis-venv"
source "$VENV/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TMPDIR=/tmp
export HF_HOME=/scratch/bgde/jhill5/hf_cache
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export VLLM_LOGGING_LEVEL=DEBUG

# ─── Paths ───
TEACHER_MODEL="/projects/bgde/jhill5/models/r1-distill-qwen-32b"
PROBLEMS="/scratch/bgde/jhill5/data/physics_problems.jsonl"
OUTPUT_DIR="/scratch/bgde/jhill5/data/physics_traces"
VLLM_PORT=8192

echo "=== JARVIS Phase 4A: Self-Distillation Trace Generation ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L | wc -l)"
echo "Model:   $TEACHER_MODEL"
echo "Problems: $(wc -l < $PROBLEMS) problems"
echo "Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Validate inputs ───
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: teacher model not found at $TEACHER_MODEL"
    exit 1
fi

if [ ! -f "$PROBLEMS" ]; then
    echo "ERROR: physics problems not found at $PROBLEMS"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" /scratch/bgde/jhill5/logs

# ─── Sanity checks ───
echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ─── Start vLLM server in background ───
VLLM_LOG="/scratch/bgde/jhill5/logs/vllm-${SLURM_JOB_ID}.log"
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

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  vLLM server ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died during startup"
        exit 1
    fi
    sleep 5
done

# Verify server is up
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start after 600s"
    echo "=== Last 100 lines of vLLM log ==="
    tail -100 "$VLLM_LOG"
    echo "=== End vLLM log ==="
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== Generating traces (8 per problem) ==="
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
    --output "/scratch/bgde/jhill5/data/physics_filtered.jsonl" \
    --target-count 5000 \
    --require-correct

echo ""
echo "=== Phase 4A Complete ==="
echo "Raw traces: $OUTPUT_DIR/traces.jsonl"
echo "Filtered traces: /scratch/bgde/jhill5/data/physics_filtered.jsonl"
