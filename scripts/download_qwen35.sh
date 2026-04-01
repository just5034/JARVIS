#!/usr/bin/env bash
# Download Qwen3.5-27B (base) and Qwen3.5-1.5B (draft) to Delta.
# Run this on a login node or inside a SLURM interactive session.
#
# Prerequisites:
#   - huggingface-cli installed (pip install huggingface_hub)
#   - HF_TOKEN set or `huggingface-cli login` done
#
# Usage:
#   bash scripts/download_qwen35.sh [--dry-run]
#
# Budget: 0 SU (download only, no GPU needed)
# Disk:   ~30 GB in /projects (base ~27 GB + draft ~3 GB)
# Time:   ~15-30 min depending on network

set -euo pipefail

MODEL_DIR="/projects/bgde/jhill5/models"
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

download() {
    local repo="$1"
    local dest="$2"
    local label="$3"

    echo "[$label] $repo -> $dest"
    if [ "$DRY_RUN" = true ]; then
        return
    fi
    mkdir -p "$dest"
    huggingface-cli download "$repo" --local-dir "$dest"
}

echo "=== Qwen3.5 Model Download for JARVIS ==="
echo "Target directory: $MODEL_DIR"
echo ""

# Check disk space
echo "--- Disk usage check ---"
df -h /projects/bgde/ 2>/dev/null || echo "(could not check /projects disk)"
echo ""

# Base model: Qwen3.5-27B
echo "--- Base Model ---"
download "Qwen/Qwen3.5-27B" \
    "$MODEL_DIR/qwen3.5-27b" \
    "Qwen3.5-27B (~27 GB, will be quantized to ~14 GB NVFP4 at serve time)"

echo ""

# Draft model: Qwen3.5-1.5B (for speculative decoding)
echo "--- Draft Model ---"
download "Qwen/Qwen3.5-1.5B" \
    "$MODEL_DIR/infrastructure/draft-model" \
    "Qwen3.5-1.5B draft (~3 GB, must match base architecture)"

echo ""

# Verify downloads
echo "--- Verification ---"
for dir in "$MODEL_DIR/qwen3.5-27b" "$MODEL_DIR/infrastructure/draft-model"; do
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo "  OK: $dir ($(du -sh $dir | cut -f1))"
    else
        echo "  MISSING: $dir"
    fi
done

echo ""
echo "=== Download complete ==="
echo ""
echo "Next steps:"
echo "  1. sbatch scripts/run_vllm_compat.sh   # Verify vLLM serves Qwen3.5"
echo "  2. sbatch scripts/run_eval_all.sh       # Run baseline evals (GPQA, AIME, LiveCodeBench)"
