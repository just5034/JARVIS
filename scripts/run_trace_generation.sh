#!/usr/bin/env bash
# Phase 4A: Generate physics reasoning traces using full DeepSeek R1-0528
# on H200 GPUs via vLLM tensor parallelism.
#
# This runs R1-0528 (W4A16 quantized, ~346GB VRAM) on 4× H200 GPUs,
# serves it via vLLM, and generates reasoning traces for physics problems.
#
# Budget: ~600-900 SU (4 GPUs × 50-75 hrs × 3 SU/GPU-hr)
#
# Prerequisites:
#   - Model downloaded: /scratch/bgde/jhill5/models/deepseek-r1-0528-w4a16
#   - Physics problems: /scratch/bgde/jhill5/data/physics_problems.jsonl
#   - Venv: /scratch/bgde/jhill5/jarvis-venv
#
# Usage:
#   sbatch scripts/run_trace_generation.sh

#SBATCH --job-name=jarvis-traces
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
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

# ─── Paths ───
TEACHER_MODEL="/scratch/bgde/jhill5/models/deepseek-r1-0528-w4a16"
PROBLEMS="/scratch/bgde/jhill5/data/physics_problems.jsonl"
OUTPUT_DIR="/scratch/bgde/jhill5/data/physics_traces"
VLLM_PORT=8192

echo "=== JARVIS Phase 4A: Trace Generation (H200) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L | wc -l)"
echo "Model:   $TEACHER_MODEL"
echo "Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Validate inputs ───
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: teacher model not found at $TEACHER_MODEL"
    exit 1
fi

if [ ! -f "$PROBLEMS" ]; then
    echo "ERROR: physics problems not found at $PROBLEMS"
    echo "Run: python -m training.data.build_physics_problems first"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" /scratch/bgde/jhill5/logs

# ─── Start vLLM server in background ───
echo "Starting vLLM server with tensor parallelism=4..."
python -m vllm.entrypoints.openai.api_server \
    --model "$TEACHER_MODEL" \
    --tensor-parallel-size 4 \
    --quantization compressed-tensors \
    --dtype float16 \
    --max-model-len 16384 \
    --port $VLLM_PORT \
    --disable-log-requests \
    &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  vLLM server ready after ${i}s"
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
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== Generating traces ==="
python -m training.physics.generate_traces_api \
    --problems "$PROBLEMS" \
    --output "$OUTPUT_DIR" \
    --model "$TEACHER_MODEL" \
    --api-base "http://localhost:$VLLM_PORT/v1" \
    --api-key "dummy" \
    --traces-per-problem 8 \
    --max-tokens 8192 \
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
python -m training.physics.rejection_sample \
    --traces "$OUTPUT_DIR/traces.jsonl" \
    --output "/scratch/bgde/jhill5/data/physics_filtered_100k.jsonl" \
    --target-count 100000 \
    --require-correct

echo ""
echo "=== Phase 4A Complete ==="
echo "Filtered traces: /scratch/bgde/jhill5/data/physics_filtered_100k.jsonl"
