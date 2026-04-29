#!/usr/bin/env bash
# Phase 4C-new: HEP code trace generation — Qwen3.5-27B as teacher,
# code_problems.jsonl as input, code-domain rejection sampling.
#
# Generates 8 traces per problem, then filters with --domain=code so
# the quality function uses syntactic-validity heuristics instead of
# the boxed-answer-based physics scorer.
#
# Budget: ~50 SU (4 GPUs × ~12 hours)
#
# Prerequisites:
#   - Model: /projects/bgde/jhill5/models/qwen3.5-27b
#   - Code problems: /work/hdd/bgde/jhill5/data/code_problems.jsonl
#       (build with: python -m training.data.build_code_problems)
#
# Usage:
#   sbatch scripts/run_code_trace_generation.sh

#SBATCH --job-name=jarvis-hep-code-traces
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
#SBATCH --output=/work/hdd/bgde/jhill5/logs/hep-code-traces-%j.out
#SBATCH --error=/work/hdd/bgde/jhill5/logs/hep-code-traces-%j.err

set -euo pipefail

# --- Environment ---
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

# --- Paths ---
TEACHER_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
PROBLEMS="/work/hdd/bgde/jhill5/data/code_problems.jsonl"
OUTPUT_DIR="/work/hdd/bgde/jhill5/data/hep_code_traces"
FILTERED="/work/hdd/bgde/jhill5/data/hep_code_filtered.jsonl"
VLLM_PORT=8193   # different port from physics job to permit concurrent runs

echo "=== JARVIS Phase 4C-new: HEP Code Trace Generation (Qwen3.5-27B) ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPUs:     $(nvidia-smi -L | wc -l)"
echo "Model:    $TEACHER_MODEL"
echo "Problems: $(wc -l < $PROBLEMS 2>/dev/null || echo 'FILE NOT FOUND') problems"
echo "Date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# --- Validate inputs ---
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: Qwen3.5-27B not found at $TEACHER_MODEL"
    echo "Run: bash scripts/download_qwen35.sh"
    exit 1
fi

if [ ! -f "$PROBLEMS" ]; then
    echo "ERROR: code problems not found at $PROBLEMS"
    echo "Generate with:"
    echo "  python -m training.data.extract_hep_code --grace-repo <PATH> --output /work/hdd/bgde/jhill5/data/hep_code_problems.jsonl"
    echo "  python -m training.data.build_code_problems"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" /work/hdd/bgde/jhill5/logs

# --- Sanity checks ---
echo "Python:  $(which python)"
echo "vLLM:    $(python -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- Start vLLM server in background ---
VLLM_LOG="/work/hdd/bgde/jhill5/logs/vllm-hep-code-traces-${SLURM_JOB_ID}.log"
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

# Wait for server (up to 30 min — Qwen3.5-27B on A100×4 typically 20-30 min cold)
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
echo "=== Generating HEP code traces (8 per problem) ==="
# Code generation needs more tokens than physics — code blocks are verbose
# (especially Geant4 macros and GDML). Bump max-tokens to 24k.
python training/physics/generate_traces_api.py \
    --problems "$PROBLEMS" \
    --output "$OUTPUT_DIR" \
    --model "$TEACHER_MODEL" \
    --api-base "http://localhost:$VLLM_PORT/v1" \
    --api-key "dummy" \
    --traces-per-problem 8 \
    --max-tokens 24576 \
    --temperature 0.7 \
    --workers 8 \
    --resume

echo ""
echo "=== Trace generation complete ==="
echo "Output: $OUTPUT_DIR"

# --- Shutdown vLLM ---
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# --- Run rejection sampling (code domain) ---
# No --require-correct: there is no ground-truth code to compare against.
# The code quality scorer uses syntactic validity of fenced blocks instead.
echo ""
echo "=== Running rejection sampling (--domain code) ==="
python training/physics/rejection_sample.py \
    --domain code \
    --traces "$OUTPUT_DIR/traces.jsonl" \
    --output "$FILTERED" \
    --target-count 2500

echo ""
echo "=== Phase 4C-new Complete ==="
echo "Raw traces:      $OUTPUT_DIR/traces.jsonl"
echo "Filtered traces: $FILTERED"
echo ""
echo "Next: sbatch scripts/run_hep_sft.sh --code"
