#!/usr/bin/env bash
# Phase 4B/4C-new: HEP LoRA training on Qwen3.5-27B.
# QDoRA (r=32) with HEP physics or HEP code data.
#
# Budget: ~200 SU per adapter (4 GPUs × ~50 hours)
#
# Prerequisites:
#   - Qwen3.5-27B: /projects/bgde/jhill5/models/qwen3.5-27b
#   - Filtered traces from Phase 4A-new
#   - Baseline evals passed
#
# Usage:
#   sbatch scripts/run_hep_sft.sh --physics    # Train hep_physics adapter
#   sbatch scripts/run_hep_sft.sh --code        # Train hep_code adapter

#SBATCH --job-name=jarvis-hep-sft
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
#SBATCH --output=/work/hdd/bgde/jhill5/logs/hep-sft-%j.out
#SBATCH --error=/work/hdd/bgde/jhill5/logs/hep-sft-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
source "$VENV/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export MASTER_PORT=29500
export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache

# ─── Parse adapter type ───
ADAPTER_TYPE=""
for arg in "$@"; do
    case $arg in
        --physics) ADAPTER_TYPE="physics" ;;
        --code)    ADAPTER_TYPE="code" ;;
        *) echo "Unknown argument: $arg. Use --physics or --code"; exit 1 ;;
    esac
done

if [ -z "$ADAPTER_TYPE" ]; then
    echo "ERROR: Specify --physics or --code"
    echo "Usage: sbatch scripts/run_hep_sft.sh --physics"
    exit 1
fi

# ─── Paths ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
TB_LOGS="/work/hdd/bgde/jhill5/tb_logs"

if [ "$ADAPTER_TYPE" = "physics" ]; then
    TRAIN_DATA="/work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl"
    OUTPUT_DIR="/work/hdd/bgde/jhill5/checkpoints/hep_physics"
    ADAPTER_NAME="hep_physics"
    EVAL_EXPERIMENT="hep_physics_sft"
elif [ "$ADAPTER_TYPE" = "code" ]; then
    TRAIN_DATA="/work/hdd/bgde/jhill5/data/hep_code_filtered.jsonl"
    OUTPUT_DIR="/work/hdd/bgde/jhill5/checkpoints/hep_code"
    ADAPTER_NAME="hep_code"
    EVAL_EXPERIMENT="hep_code_sft"
fi

echo "=== JARVIS HEP LoRA Training — ${ADAPTER_NAME} ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPUs:       $(nvidia-smi -L | wc -l)"
echo "Base model: $BASE_MODEL"
echo "Adapter:    $ADAPTER_NAME"
echo "Train data: $TRAIN_DATA"
echo "Output:     $OUTPUT_DIR"
echo "Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Validate inputs ───
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Qwen3.5-27B not found at $BASE_MODEL"
    echo "Run: bash scripts/download_qwen35.sh"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: training data not found at $TRAIN_DATA"
    echo "Run Phase 4A-new first (sbatch scripts/run_trace_generation.sh)"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$TB_LOGS"

# ─── Copy data to local SSD for speed ───
echo "Copying training data to local SSD..."
cp "$TRAIN_DATA" /tmp/train_data.jsonl
LOCAL_TRAIN_DATA=/tmp/train_data.jsonl
echo "  $(wc -l < $LOCAL_TRAIN_DATA) training examples"
echo ""

# ─── Run SFT with DeepSpeed ZeRO-3 ───
# Qwen3.5-27B at bf16 = ~54 GB, needs ZeRO-3 across 4× A100-40GB.
# QDoRA = Quantized DoRA (rank-32), targets all linear layers.

deepspeed --num_gpus=4 --module training.physics.run_sft \
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
    --logging_dir "$TB_LOGS/${ADAPTER_NAME}_sft"

echo ""
echo "=== SFT Complete ==="
echo "Checkpoints: $OUTPUT_DIR"
echo ""

# ─── Copy final adapter to persistent storage ───
FINAL_ADAPTER="/projects/bgde/jhill5/adapters/${ADAPTER_NAME}"
if [ -d "$OUTPUT_DIR/final" ]; then
    echo "Copying final adapter to $FINAL_ADAPTER..."
    mkdir -p "$FINAL_ADAPTER"
    cp -r "$OUTPUT_DIR/final/"* "$FINAL_ADAPTER/"
    echo "  Done."
elif [ -d "$OUTPUT_DIR/checkpoint-$(ls $OUTPUT_DIR | grep checkpoint | sort -t- -k2 -n | tail -1 | cut -d- -f2)" ]; then
    LAST_CKPT="$OUTPUT_DIR/$(ls $OUTPUT_DIR | grep checkpoint | sort -t- -k2 -n | tail -1)"
    echo "Copying last checkpoint to $FINAL_ADAPTER..."
    mkdir -p "$FINAL_ADAPTER"
    cp -r "$LAST_CKPT/"* "$FINAL_ADAPTER/"
    echo "  Done."
fi
echo ""

# ─── Post-SFT eval ───
echo "=== Running post-SFT GPQA eval ==="
python -m training.eval.run_gpqa \
    --model "$BASE_MODEL" \
    --adapter "$FINAL_ADAPTER" \
    --output "/work/hdd/bgde/jhill5/eval/gpqa_${ADAPTER_NAME}_$(date +%Y%m%d).json" \
    --data-dir "/work/hdd/bgde/jhill5/data/benchmarks" \
    --log-dir "$TB_LOGS" \
    --experiment "$EVAL_EXPERIMENT"

echo ""
echo "=== Phase 4B/4C — ${ADAPTER_NAME} Training Complete ==="
echo "Adapter saved to: $FINAL_ADAPTER"
echo ""
echo "To view training curves:"
echo "  ssh -L 6006:localhost:6006 jhill5@login.delta.ncsa.illinois.edu"
echo "  tensorboard --logdir $TB_LOGS --port 6006"
