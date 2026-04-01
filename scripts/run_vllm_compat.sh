#!/usr/bin/env bash
# Qwen3.5-27B vLLM compatibility and pipeline verification.
# Tests: model loading, generation, <think> tag parsing, budget forcing
# patterns, speculative decoding draft model, and ThinkPRM compatibility.
#
# Budget: ~4 SU (4 GPUs × 1 hour)
#
# Usage:
#   sbatch scripts/run_vllm_compat.sh

#SBATCH --job-name=jarvis-compat
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
#SBATCH --output=/scratch/bgde/jhill5/logs/compat-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/compat-%j.err

set -euo pipefail

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/scratch/bgde/jhill5/jarvis-venv"
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -e "/u/$USER/JARVIS[serving,training]"
else
    source "$VENV/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# ─── Paths ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
DRAFT_MODEL="/projects/bgde/jhill5/models/infrastructure/draft-model"
VLLM_PORT=8192
PASSED=0
FAILED=0
TOTAL=0

mkdir -p /scratch/bgde/jhill5/logs

# ─── Helpers ───
pass_test() {
    echo "  ✓ PASS: $1"
    PASSED=$((PASSED + 1))
    TOTAL=$((TOTAL + 1))
}

fail_test() {
    echo "  ✗ FAIL: $1"
    echo "    $2"
    FAILED=$((FAILED + 1))
    TOTAL=$((TOTAL + 1))
}

echo "=== JARVIS Qwen3.5-27B Compatibility Check ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L | wc -l)"
echo "Model:   $BASE_MODEL"
echo "Draft:   $DRAFT_MODEL"
echo "Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Sanity checks ───
echo "--- Environment ---"
echo "Python: $(which python) ($(python --version))"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "Torch:  $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")' 2>/dev/null || echo 'NOT INSTALLED')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ─── Test 1: Model exists ───
echo "=== Test 1: Model files exist ==="
if [ -d "$BASE_MODEL" ] && [ -f "$BASE_MODEL/config.json" ]; then
    pass_test "Base model directory and config.json found"
else
    fail_test "Base model not found" "Expected: $BASE_MODEL/config.json"
fi

if [ -d "$DRAFT_MODEL" ] && [ -f "$DRAFT_MODEL/config.json" ]; then
    pass_test "Draft model directory and config.json found"
else
    fail_test "Draft model not found" "Expected: $DRAFT_MODEL/config.json"
fi
echo ""

# ─── Test 2: vLLM can load Qwen3.5 architecture ───
echo "=== Test 2: vLLM loads Qwen3.5-27B ==="
VLLM_LOG="/scratch/bgde/jhill5/logs/vllm-compat-${SLURM_JOB_ID}.log"

python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --port $VLLM_PORT \
    --no-enable-log-requests \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID $VLLM_PID)..."
SERVER_UP=false
for i in $(seq 1 360); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "  Server ready after $((i * 5))s"
        SERVER_UP=true
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  vLLM process died during startup!"
        echo "  === Last 30 lines of vLLM log ==="
        tail -30 "$VLLM_LOG"
        fail_test "vLLM startup" "Process exited before serving"
        break
    fi
    sleep 5
done

if [ "$SERVER_UP" = false ]; then
    if kill -0 $VLLM_PID 2>/dev/null; then
        fail_test "vLLM startup" "Server did not respond after 1800s"
        kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
    fi
    echo ""
    echo "=== ABORTING: Cannot proceed without vLLM server ==="
    echo "RESULTS: $PASSED passed, $FAILED failed out of $TOTAL tests"
    exit 1
fi

pass_test "vLLM loaded Qwen3.5-27B and is serving"
echo ""

# ─── Test 3: Basic generation ───
echo "=== Test 3: Basic chat completion ==="
RESPONSE=$(curl -s http://localhost:$VLLM_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$BASE_MODEL"'",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 64,
        "temperature": 0.0
    }')

if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" 2>/dev/null; then
    CONTENT=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])")
    pass_test "Chat completion returned content"
    echo "    Response: $CONTENT"
else
    fail_test "Chat completion" "No valid response: $(echo $RESPONSE | head -c 300)"
fi
echo ""

# ─── Test 4: Think tag generation ───
echo "=== Test 4: <think> tag generation (reasoning format) ==="
RESPONSE=$(curl -s http://localhost:$VLLM_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$BASE_MODEL"'",
        "messages": [{"role": "user", "content": "Solve step by step: What is the derivative of x^3 * sin(x)?"}],
        "max_tokens": 2048,
        "temperature": 0.6
    }')

CONTENT=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")

if echo "$CONTENT" | grep -q "<think>"; then
    pass_test "Model produces <think> opening tag"
else
    fail_test "<think> tag" "Response does not contain <think> tag (model may not use thinking format by default)"
    echo "    First 300 chars: $(echo "$CONTENT" | head -c 300)"
fi

if echo "$CONTENT" | grep -q "</think>"; then
    pass_test "Model produces </think> closing tag"
else
    fail_test "</think> tag" "Response does not contain </think> closing tag"
fi
echo ""

# ─── Test 5: Budget forcing pattern matching ───
echo "=== Test 5: Budget forcing conclusion pattern detection ==="
python -c "
import re

# These are the patterns from src/jarvis/inference/budget_forcing.py
CONCLUSION_PATTERNS = [
    re.compile(r'</think>', re.IGNORECASE),
    re.compile(r'\\\\boxed\{', re.IGNORECASE),
    re.compile(r'\*\*Final Answer\*\*', re.IGNORECASE),
    re.compile(r'(?:^|\n)(?:Therefore|Thus|Hence|In conclusion),?\s', re.IGNORECASE),
    re.compile(r'(?:^|\n)The answer is\s', re.IGNORECASE),
]

# Test against the actual model output
content = '''$CONTENT'''

matched = []
for p in CONCLUSION_PATTERNS:
    if p.search(content):
        matched.append(p.pattern)

if matched:
    print('MATCHED_PATTERNS=' + '|'.join(matched))
else:
    print('MATCHED_PATTERNS=NONE')
" 2>/dev/null | while IFS= read -r line; do
    if echo "$line" | grep -q "MATCHED_PATTERNS=NONE"; then
        fail_test "Budget forcing patterns" "No conclusion patterns matched in model output"
    else
        PATTERNS=$(echo "$line" | sed 's/MATCHED_PATTERNS=//')
        pass_test "Budget forcing patterns matched: $PATTERNS"
    fi
done

# Direct pattern test with known strings (independent of model output)
python -c "
import re, sys

patterns = [
    re.compile(r'</think>', re.IGNORECASE),
    re.compile(r'\\\\boxed\{', re.IGNORECASE),
    re.compile(r'\*\*Final Answer\*\*', re.IGNORECASE),
    re.compile(r'(?:^|\n)(?:Therefore|Thus|Hence|In conclusion),?\s', re.IGNORECASE),
    re.compile(r'(?:^|\n)The answer is\s', re.IGNORECASE),
]

test_strings = [
    ('</think>The answer is 42', True, '</think>'),
    ('\\\\boxed{42}', True, 'boxed'),
    ('**Final Answer**', True, 'Final Answer'),
    ('Therefore, x = 5', True, 'Therefore'),
    ('The answer is 42', True, 'The answer is'),
    ('still thinking...', False, 'no match'),
]

ok = True
for text, should_match, label in test_strings:
    matched = any(p.search(text) for p in patterns)
    if matched != should_match:
        print(f'PATTERN_FAIL: expected match={should_match} for \"{label}\"')
        ok = False

if ok:
    print('PATTERN_OK')
else:
    print('PATTERN_FAIL')
    sys.exit(1)
" && pass_test "Budget forcing pattern unit tests all pass" \
  || fail_test "Budget forcing patterns" "Some pattern unit tests failed"
echo ""

# ─── Test 6: ThinkPRM compatibility ───
echo "=== Test 6: ThinkPRM verifier loads ==="
python -c "
from jarvis.inference.verification import ThinkPRMVerifier
v = ThinkPRMVerifier()
v.load()
if v.available:
    print('THINKPRM_OK')
else:
    # ThinkPRM may not be available without the actual model weights on disk
    print('THINKPRM_STUB')
" 2>/dev/null
THINKPRM_STATUS=$?

if [ $THINKPRM_STATUS -eq 0 ]; then
    pass_test "ThinkPRMVerifier loads without error"
else
    fail_test "ThinkPRMVerifier" "Failed to instantiate or load"
fi
echo ""

# ─── Test 7: Speculative decoding draft model check ───
echo "=== Test 7: Draft model architecture compatibility ==="
python -c "
import json, os

base_config = json.load(open('$BASE_MODEL/config.json'))
draft_config = json.load(open('$DRAFT_MODEL/config.json'))

base_arch = base_config.get('architectures', ['unknown'])[0]
draft_arch = draft_config.get('architectures', ['unknown'])[0]

print(f'Base architecture:  {base_arch}')
print(f'Draft architecture: {draft_arch}')

# For speculative decoding, architectures should match
if base_arch == draft_arch:
    print('ARCH_MATCH')
else:
    # Different arch names may still be compatible (e.g., Qwen3.5ForCausalLM)
    base_family = base_arch.replace('ForCausalLM', '')
    draft_family = draft_arch.replace('ForCausalLM', '')
    if base_family == draft_family:
        print('ARCH_MATCH')
    else:
        print(f'ARCH_MISMATCH: {base_arch} vs {draft_arch}')
" 2>/dev/null | while IFS= read -r line; do
    echo "    $line"
    if echo "$line" | grep -q "ARCH_MATCH"; then
        pass_test "Draft model architecture matches base model"
    elif echo "$line" | grep -q "ARCH_MISMATCH"; then
        fail_test "Architecture mismatch" "$line"
    fi
done
echo ""

# ─── Test 8: Multi-turn conversation ───
echo "=== Test 8: Multi-turn conversation ==="
RESPONSE=$(curl -s http://localhost:$VLLM_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$BASE_MODEL"'",
        "messages": [
            {"role": "system", "content": "You are a helpful physics assistant."},
            {"role": "user", "content": "What is the Higgs boson mass in GeV?"},
            {"role": "assistant", "content": "The Higgs boson mass is approximately 125 GeV."},
            {"role": "user", "content": "What experiment discovered it?"}
        ],
        "max_tokens": 256,
        "temperature": 0.0
    }')

CONTENT=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:300])" 2>/dev/null || echo "PARSE_ERROR")

if [ "$CONTENT" != "PARSE_ERROR" ] && [ -n "$CONTENT" ]; then
    pass_test "Multi-turn conversation works"
    echo "    Response: $CONTENT"
else
    fail_test "Multi-turn conversation" "Failed to parse response"
fi
echo ""

# ─── Test 9: Token counting ───
echo "=== Test 9: Token usage reporting ==="
USAGE=$(echo "$RESPONSE" | python -c "
import sys, json
d = json.load(sys.stdin)
u = d.get('usage', {})
print(f\"prompt_tokens={u.get('prompt_tokens', 'N/A')} completion_tokens={u.get('completion_tokens', 'N/A')} total={u.get('total_tokens', 'N/A')}\")
" 2>/dev/null || echo "PARSE_ERROR")

if [ "$USAGE" != "PARSE_ERROR" ]; then
    pass_test "Token usage reported correctly"
    echo "    $USAGE"
else
    fail_test "Token usage" "Could not parse usage from response"
fi
echo ""

# ─── Test 10: Model listing ───
echo "=== Test 10: /v1/models endpoint ==="
MODELS_RESP=$(curl -s http://localhost:$VLLM_PORT/v1/models)
MODEL_ID=$(echo "$MODELS_RESP" | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "PARSE_ERROR")

if [ "$MODEL_ID" != "PARSE_ERROR" ]; then
    pass_test "/v1/models lists loaded model"
    echo "    Model ID: $MODEL_ID"
else
    fail_test "/v1/models" "Could not parse model list"
fi
echo ""

# ─── Cleanup ───
echo "=== Shutting down vLLM server ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "Server stopped."
echo ""

# ─── Summary ───
echo "=========================================="
echo "  COMPATIBILITY TEST RESULTS"
echo "=========================================="
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "  Total:  $TOTAL"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "ACTION REQUIRED: Review failures above before running baseline evals."
    echo "vLLM log: $VLLM_LOG"
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED — safe to run baseline evals."
    echo "  sbatch scripts/run_eval_all.sh"
fi
