#!/usr/bin/env bash
# Phase 4B: Physics brain distillation SFT on Delta.
# QDoRA (r=32) on R1-Distill-Qwen-32B with filtered traces.
# Budget: ~800 SU (4 GPUs × ~200 hours)
#
# Prerequisites:
#   - Traces generated and filtered (Phase 4A)
#   - Base model downloaded to /projects/
#
# Usage:
#   sbatch scripts/run_physics_sft.sh

#SBATCH --job-name=jarvis-physics-sft
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde-delta-gpu/logs/physics-sft-%j.out
#SBATCH --error=/scratch/bgde-delta-gpu/logs/physics-sft-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde-delta-gpu/jarvis-venv"
source "$VENV/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export MASTER_PORT=29500
export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache

# ─── Paths ───
BASE_MODEL="/projects/bgde-delta-gpu/models/r1-distill-qwen-32b"
TRAIN_DATA="/scratch/bgde-delta-gpu/data/physics_filtered_100k.jsonl"
OUTPUT_DIR="/scratch/bgde-delta-gpu/checkpoints/physics_sft"
TB_LOGS="/scratch/bgde-delta-gpu/tb_logs"

echo "=== JARVIS Physics SFT (Phase 4B) ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi -L | wc -l)"
echo "Base model: $BASE_MODEL"
echo "Train data: $TRAIN_DATA"
echo "Output:     $OUTPUT_DIR"
echo "Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Validate inputs ───
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: base model not found at $BASE_MODEL"
    echo "Run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', local_dir='$BASE_MODEL')\""
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: training data not found at $TRAIN_DATA"
    echo "Run Phase 4A first (generate_traces_api.py → rejection_sample.py)"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$(dirname $OUTPUT_DIR)/../logs"

# ─── Copy data to local SSD for speed ───
echo "Copying training data to local SSD..."
cp "$TRAIN_DATA" /tmp/train_data.jsonl
LOCAL_TRAIN_DATA=/tmp/train_data.jsonl

# ─── Run SFT with DeepSpeed ZeRO-3 ───
# ZeRO-3 is REQUIRED: 32B model at fp16 = ~64 GB, exceeds single A100 40 GB.
# QDoRA = Quantized DoRA (rank-32), targets all linear layers.

deepspeed --num_gpus=4 -m training.physics.run_sft \
    --model_name_or_path "$BASE_MODEL" \
    --train_data "$LOCAL_TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed configs/ds_zero3.json \
    --use_dora true \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 8192 \
    --bf16 true \
    --save_strategy epoch \
    --logging_steps 10 \
    --logging_dir "$TB_LOGS/physics_sft"

echo ""
echo "=== SFT Complete ==="
echo "Checkpoints: $OUTPUT_DIR"
echo ""

# ─── Quick eval after SFT ───
echo "=== Running post-SFT GPQA eval ==="
python -m training.eval.run_gpqa \
    --model "$BASE_MODEL" \
    --adapter "$OUTPUT_DIR/final" \
    --output "/scratch/bgde-delta-gpu/eval/gpqa_post_sft_$(date +%Y%m%d).json" \
    --data-dir "/scratch/bgde-delta-gpu/data/benchmarks" \
    --log-dir "$TB_LOGS" \
    --experiment "physics_sft"

echo "=== Phase 4B Done ==="
