#!/usr/bin/env bash
# Quick probe: does vLLM + Qwen3.5 tool calling work?
# Starts vLLM, makes ONE tool call, prints result, exits.
# Should take ~20 min total (mostly vLLM startup).
#
# Usage: sbatch scripts/probe_toolcall.sh

#SBATCH --job-name=jarvis-probe
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/probe-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/probe-%j.err

set -euo pipefail

module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8
source /scratch/bgde/jhill5/jarvis-venv/bin/activate

export HF_HOME=/tmp/hf_cache
BASE_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"
PORT=8194

echo "=== Tool Call Probe ==="
echo "Testing: --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
echo ""

# Start vLLM with tool calling
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
    > /scratch/bgde/jhill5/logs/probe-vllm-${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM (PID $VLLM_PID)..."
WAITED=0
READY=0
while [ $WAITED -lt 2700 ]; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (${WAITED}s)"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FATAL: vLLM died during startup"
        echo "Check: /scratch/bgde/jhill5/logs/probe-vllm-${SLURM_JOB_ID}.log"
        tail -30 /scratch/bgde/jhill5/logs/probe-vllm-${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done

if [ $READY -ne 1 ]; then
    echo "FATAL: vLLM did not become ready within ${WAITED}s"
    echo "Last 40 lines of vLLM log:"
    tail -40 /scratch/bgde/jhill5/logs/probe-vllm-${SLURM_JOB_ID}.log
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== Test 1: Tool call (should return TOOL_CALL_OK) ==="
python -c "
from openai import OpenAI
c = OpenAI(api_key='x', base_url='http://localhost:${PORT}/v1')
r = c.chat.completions.create(
    model='${BASE_MODEL}',
    messages=[{'role':'user','content':'List the files in the current directory.'}],
    tools=[{
        'type':'function',
        'function':{
            'name':'list_dir',
            'description':'List directory contents',
            'parameters':{'type':'object','properties':{'path':{'type':'string','description':'directory path'}},'required':['path']}
        }
    }],
    max_tokens=1024,
    temperature=0.6,
    top_p=0.95,
)
tc = r.choices[0].message.tool_calls
content = r.choices[0].message.content or ''
print(f'finish_reason: {r.choices[0].finish_reason}')
if tc:
    for t in tc:
        print(f'TOOL_CALL_OK: {t.function.name}({t.function.arguments})')
else:
    print(f'NO_TOOL_CALL: got text ({len(content)} chars)')
    print(f'  text: {content[:500]}')
"

echo ""
echo "=== Test 2: Multi-turn tool use (tool result round-trip) ==="
python -c "
from openai import OpenAI
import json
c = OpenAI(api_key='x', base_url='http://localhost:${PORT}/v1')
tools = [{
    'type':'function',
    'function':{
        'name':'read_file',
        'description':'Read a file and return its contents',
        'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}
    }
},{
    'type':'function',
    'function':{
        'name':'finish',
        'description':'Signal completion',
        'parameters':{'type':'object','properties':{},'required':[]}
    }
}]

# Turn 1: ask model to read a file
msgs = [
    {'role':'system','content':'You fix bugs. Use read_file to inspect code, then finish.'},
    {'role':'user','content':'Read the file setup.py and then call finish.'},
]
r = c.chat.completions.create(model='${BASE_MODEL}', messages=msgs, tools=tools, max_tokens=1024, temperature=0.6)
tc = r.choices[0].message.tool_calls
if not tc:
    print(f'Turn 1 FAIL: no tool call. Got: {(r.choices[0].message.content or \"\")[:200]}')
else:
    print(f'Turn 1 OK: {tc[0].function.name}({tc[0].function.arguments})')

    # Turn 2: send tool result, expect finish()
    msgs.append({'role':'assistant','content':r.choices[0].message.content or '','tool_calls':[{'id':tc[0].id,'type':'function','function':{'name':tc[0].function.name,'arguments':tc[0].function.arguments}}]})
    msgs.append({'role':'tool','tool_call_id':tc[0].id,'content':'# setup.py\\nfrom setuptools import setup\\nsetup(name=\"example\")'})
    r2 = c.chat.completions.create(model='${BASE_MODEL}', messages=msgs, tools=tools, max_tokens=1024, temperature=0.6)
    tc2 = r2.choices[0].message.tool_calls
    if tc2:
        print(f'Turn 2 OK: {tc2[0].function.name}({tc2[0].function.arguments})')
    else:
        print(f'Turn 2: text response ({len(r2.choices[0].message.content or \"\")} chars): {(r2.choices[0].message.content or \"\")[:200]}')
"

echo ""
echo "=== Probe complete ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
