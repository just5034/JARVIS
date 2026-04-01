#!/usr/bin/env bash
# Download all JARVIS model weights from HuggingFace.
# Usage: ./scripts/download_models.sh [--dry-run] [--skip-optional]

set -euo pipefail

MODEL_DIR="${JARVIS_MODEL_DIR:-/models}"
DRY_RUN=false
SKIP_OPTIONAL=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --skip-optional) SKIP_OPTIONAL=true ;;
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

echo "=== JARVIS Model Download ==="
echo "Target directory: $MODEL_DIR"
echo ""

# Core base model
echo "--- Base Model ---"
download "Qwen/Qwen3.5-27B" \
    "$MODEL_DIR/brains/qwen3.5-27b" \
    "Unified base (Qwen3.5, ~27GB raw, ~14GB FP4)"

# Infrastructure
echo ""
echo "--- Infrastructure ---"
download "Qwen/Qwen3.5-0.8B" \
    "$MODEL_DIR/infrastructure/draft-model" \
    "Draft model for speculative decoding (~1.6GB, must match Qwen3.5 arch)"

download "PRIME-RL/ThinkPRM-1.5B" \
    "$MODEL_DIR/infrastructure/think-prm" \
    "ThinkPRM verifier (~0.8GB)"

download "sentence-transformers/all-MiniLM-L6-v2" \
    "$MODEL_DIR/infrastructure/rag-embedding" \
    "RAG embedding model (~0.09GB)"

# Specialists
echo ""
echo "--- Specialists ---"
download "AI4Chem/ChemLLM-7B-Chat" \
    "$MODEL_DIR/specialists/chemllm-7b" \
    "Chemistry specialist (~3.5GB FP4)"

download "BioMistral/BioMistral-7B" \
    "$MODEL_DIR/specialists/biomistral-7b" \
    "Biology specialist (~3.5GB FP4)"

download "EvolutionaryScale/esm3-sm-open-v1" \
    "$MODEL_DIR/specialists/esm3-open" \
    "Protein specialist (~0.7GB FP16)"

download "arcinstitute/evo2-7b" \
    "$MODEL_DIR/specialists/evo2-7b" \
    "Genomics specialist (~3.5GB FP4)"

echo ""
echo "=== Download complete ==="
