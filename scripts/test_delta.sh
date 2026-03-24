#!/usr/bin/env bash
# SLURM job script to test JARVIS Phase 1 on Delta with a small model.
#
# Usage:
#   sbatch scripts/test_delta.sh
#
# This loads Qwen2.5-7B-Instruct (small enough for a single A100-40GB)
# and runs a quick smoke test against the API.

#SBATCH --job-name=jarvis-test
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/jarvis-test-%j.out
#SBATCH --error=logs/jarvis-test-%j.err

set -euo pipefail

# --- Environment setup ---
module purge
module load gcc
module load cuda

cd "$HOME/JARVIS"

# Create venv if it doesn't exist
VENV_DIR="$HOME/JARVIS/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install JARVIS with serving deps
pip install -e ".[serving,dev]" --quiet

# --- Model setup ---
# Use a small model for testing — Qwen2.5-7B fits easily on A100-40GB
# On first run, this downloads from HuggingFace (~15GB)
export HF_HOME=/tmp/hf_cache
export JARVIS_MODEL_DIR=/tmp/jarvis_models

# Create a test-specific models.yaml that uses a 7B model
TEST_CONFIG_DIR="/tmp/jarvis_test_configs"
mkdir -p "$TEST_CONFIG_DIR"

# Copy base configs
cp configs/inference.yaml "$TEST_CONFIG_DIR/"
cp configs/router.yaml "$TEST_CONFIG_DIR/"

# Override models.yaml with a small test model
cat > "$TEST_CONFIG_DIR/models.yaml" << 'YAML'
base_models:
  qwen2_5_7b:
    model_id: "Qwen/Qwen2.5-7B-Instruct"
    architecture: "qwen2.5"
    path: "/tmp/jarvis_models/qwen2.5-7b"
    size_gb: 4.0
    quantization: "none"
    context_length: 32768
    recommended_max_context: 8192
    load_policy: "always_resident"
    roles: ["general"]

lora_adapters: {}

infrastructure:
  router:
    model_id: "bert-base-uncased"
    path: "/models/infrastructure/router"
    size_gb: 0.06
    load_policy: "always_resident"
  think_prm:
    model_id: "PRIME-RL/ThinkPRM-1.5B"
    path: "/models/infrastructure/think-prm"
    size_gb: 0.8
    load_policy: "always_resident"
  draft_model:
    model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    path: "/models/infrastructure/draft-model"
    size_gb: 0.8
    load_policy: "always_resident"
  rag_embedding:
    model_id: "sentence-transformers/all-MiniLM-L6-v2"
    path: "/models/infrastructure/rag-embedding"
    size_gb: 0.09
    load_policy: "always_resident"

specialists: {}
YAML

# Override deployment.yaml for test
cat > "$TEST_CONFIG_DIR/deployment.yaml" << 'YAML'
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1

memory_budget:
  total_gb: 40
  reserved_os_gb: 2
  reserved_framework_gb: 4
  safety_margin_gb: 2

logging:
  level: "INFO"
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

model_dir: "/tmp/jarvis_models"
YAML

# --- Validate configs ---
echo "=== Validating configs ==="
python -m jarvis --config "$TEST_CONFIG_DIR" validate

# --- Start server in background ---
echo "=== Starting JARVIS server ==="
mkdir -p logs
python -m jarvis --config "$TEST_CONFIG_DIR" serve \
    --load-model qwen2_5_7b \
    --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 180); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process exited"
        cat logs/jarvis-test-${SLURM_JOB_ID}.err 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# --- Smoke tests ---
echo ""
echo "=== Running smoke tests ==="

echo "--- Health check ---"
curl -s http://localhost:8000/health | python -m json.tool

echo ""
echo "--- List models ---"
curl -s http://localhost:8000/v1/models | python -m json.tool

echo ""
echo "--- Memory status ---"
curl -s http://localhost:8000/admin/memory | python -m json.tool

echo ""
echo "--- Chat completion (non-streaming) ---"
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful physics assistant."},
            {"role": "user", "content": "What is the mass of the Higgs boson in GeV?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }' | python -m json.tool

echo ""
echo "--- Chat completion (streaming) ---"
curl -s -N -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Write a one-line Python function to compute factorial."}
        ],
        "temperature": 0.3,
        "max_tokens": 128,
        "stream": true
    }'

echo ""
echo ""
echo "=== All smoke tests complete ==="

# --- Cleanup ---
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Server stopped"
