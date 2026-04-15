#!/usr/bin/env bash
# Run SWE-bench Verified prediction generation against Qwen3.5-27B on Delta.
#
# First run: N=20 instances to validate the pipeline (~3-5 hours)
# Full run:  remove --n-instances flag (~80-100 hours, split into chunks)
#
# Output: predictions JSON. Scoring is a SEPARATE step requiring Docker.
#
# Usage:
#   sbatch scripts/run_swebench.sh                # default N=20 baseline
#   sbatch scripts/run_swebench.sh --n 50         # custom N
#   sbatch scripts/run_swebench.sh --resume       # resume from existing predictions

#SBATCH --job-name=jarvis-swebench
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --constraint="projects"
#SBATCH --output=/work/hdd/bgde/jhill5/logs/swebench-%j.out
#SBATCH --error=/work/hdd/bgde/jhill5/logs/swebench-%j.err

set -euo pipefail

# ─── Args ───
N_INSTANCES=20
RESUME=false
for arg in "$@"; do
    case $arg in
        --n=*) N_INSTANCES="${arg#*=}" ;;
        --n)   shift; N_INSTANCES="$1" ;;
        --resume) RESUME=true ;;
    esac
done

# ─── Environment ───
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

VENV="/work/hdd/bgde/jhill5/jarvis-venv"
source "$VENV/bin/activate"

# Make sure openai client is installed (used by the agent)
pip install --quiet openai datasets 2>&1 | tail -5 || true

export HF_HOME=/work/hdd/bgde/jhill5/hf_cache
export TMPDIR=/tmp

# ─── Paths ───
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
EVAL_OUT="/work/hdd/bgde/jhill5/eval"
# Repo clones are ephemeral — keep them on node-local scratch.
# Override with SWEBENCH_WORKDIR=... sbatch ... if needed.
WORKDIR="${SWEBENCH_WORKDIR:-${TMPDIR:-/tmp}/swebench_workspaces}"
PORT=8193  # different from ARIA's 8192 to avoid clashes
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$EVAL_OUT" "$WORKDIR" /work/hdd/bgde/jhill5/logs

# Determine output path (resume uses the latest existing file)
if $RESUME; then
    LATEST=$(ls -t "$EVAL_OUT"/swebench_qwen35_*.json 2>/dev/null | head -1 || true)
    if [ -n "$LATEST" ]; then
        OUTPUT="$LATEST"
        echo "Resuming from $OUTPUT"
    else
        OUTPUT="$EVAL_OUT/swebench_qwen35_${TIMESTAMP}.json"
        echo "No existing predictions found, starting fresh"
    fi
else
    OUTPUT="$EVAL_OUT/swebench_qwen35_${TIMESTAMP}.json"
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Model not found at $BASE_MODEL"
    exit 1
fi

echo "=== SWE-bench Verified Eval — Qwen3.5-27B ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L | wc -l)"
echo "Model:     $BASE_MODEL"
echo "N instances: $N_INSTANCES"
echo "Output:    $OUTPUT"
echo "Workdir:   $WORKDIR"
echo "Date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ─── Start vLLM server WITH tool calling support ───
# Critical flags for Qwen3.5 tool calling:
#   --enable-auto-tool-choice  : allows tools= param in API calls
#   --tool-call-parser qwen3_coder : parses <tool_call> tags from Qwen3.5 output
#   --reasoning-parser qwen3 : strips <think>...</think> before parsing tool calls
# Without these, the tools param is silently ignored and the model outputs plain text.
echo "Starting vLLM server with tool calling on port $PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port $PORT \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    > /work/hdd/bgde/jhill5/logs/swebench-vllm-${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ─── Wait for vLLM ───
echo "Waiting for vLLM (~15-20 min for Qwen3.5)..."
MAX_WAIT=1800
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died — check /work/hdd/bgde/jhill5/logs/swebench-vllm-${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ─── Verify tool calling works before running full eval ───
echo "Probing tool calling support..."
PROBE_RESULT=$(python -c "
from openai import OpenAI
c = OpenAI(api_key='x', base_url='http://localhost:${PORT}/v1')
try:
    r = c.chat.completions.create(
        model='$BASE_MODEL',
        messages=[{'role':'user','content':'What is 2+2? Use the calculator tool.'}],
        tools=[{'type':'function','function':{'name':'calc','description':'calculator','parameters':{'type':'object','properties':{'expr':{'type':'string'}},'required':['expr']}}}],
        max_tokens=512,
        temperature=0.6,
    )
    tc = r.choices[0].message.tool_calls
    if tc and len(tc) > 0:
        print(f'TOOL_CALL_OK: {tc[0].function.name}({tc[0].function.arguments})')
    else:
        content = r.choices[0].message.content or ''
        print(f'NO_TOOL_CALL: model returned text ({len(content)} chars): {content[:200]}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "  Probe result: $PROBE_RESULT"

if echo "$PROBE_RESULT" | grep -q "TOOL_CALL_OK"; then
    echo "  Tool calling is working!"
elif echo "$PROBE_RESULT" | grep -q "ERROR"; then
    echo "  FATAL: Tool calling failed. Check vLLM flags (--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3)"
    echo "  vLLM log: /work/hdd/bgde/jhill5/logs/swebench-vllm-${SLURM_JOB_ID}.log"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
else
    echo "  WARNING: Model did not use tool calling. It may work on real problems but tool call parsing may be unreliable."
    echo "  Continuing anyway..."
fi

# ─── Run prediction generation ───
echo ""
echo "=== Running SWE-bench prediction agent ==="
echo "Sampling: temp=0.6, top_p=0.95, max_tokens=4096"
echo ""

EXTRA_ARGS=""
if [ "$N_INSTANCES" != "all" ]; then
    EXTRA_ARGS="--n-instances $N_INSTANCES"
fi

python -m training.eval.run_swebench \
    --base-url "http://localhost:${PORT}/v1" \
    --model "$BASE_MODEL" \
    --api-key "not-needed" \
    --output "$OUTPUT" \
    --workdir "$WORKDIR" \
    --max-steps 25 \
    --max-tokens 4096 \
    --temperature 0.6 \
    $EXTRA_ARGS

# ─── Cleanup ───
echo ""
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "  SWE-BENCH PREDICTIONS COMPLETE"
echo "=========================================="
echo "Predictions: $OUTPUT"
echo ""
echo "Quick stats:"
python -c "
import json
with open('$OUTPUT') as f:
    preds = json.load(f)
total = len(preds)
finished = sum(1 for p in preds if p.get('finished'))
empty = sum(1 for p in preds if not p.get('model_patch','').strip())
errored = sum(1 for p in preds if p.get('error'))
print(f'  Total predictions:  {total}')
print(f'  Finished cleanly:   {finished}')
print(f'  Empty patches:      {empty}')
print(f'  Errored:            {errored}')
" 2>/dev/null || echo "  (could not compute stats)"
echo ""
echo "Next step: score predictions with the official swebench harness:"
echo "  pip install swebench"
echo "  python -m swebench.harness.run_evaluation \\"
echo "      --predictions_path $OUTPUT \\"
echo "      --max_workers 4 --run_id qwen35_baseline \\"
echo "      --dataset_name princeton-nlp/SWE-bench_Verified"
echo "(Requires Docker; can run on a separate machine.)"
