# CodeAgent + JARVIS Integration Gap

## The Goal

Run the CodeAgent clone (D:\jarvis-body) using JARVIS as its LLM brain instead of the Anthropic API. CodeAgent is a production coding agent — it reads files, edits code, runs bash commands, searches codebases — all autonomously.

## How CodeAgent Works

CodeAgent operates in a **tool-call loop**:

```
1. CodeAgent sends a message to the LLM: "The user wants to fix bug X"
2. LLM responds: "I need to read the file. Call tool: read_file(path='/src/app.py')"
3. CodeAgent executes the tool, sends the result back to the LLM
4. LLM responds: "I see the bug. Call tool: edit_file(path='/src/app.py', ...)"
5. CodeAgent executes the edit, sends confirmation back
6. LLM responds: "Done. Here's what I fixed: ..."
```

**Every action the agent takes** — every file read, every edit, every bash command, every search — goes through this tool-call loop. There is no text-only fallback. If the LLM can't emit structured tool calls, nothing works.

## The Two Mismatches

### Mismatch 1: API Format (Mechanical — Easy to Fix)

CodeAgent speaks the **Anthropic Messages API**. JARVIS speaks the **OpenAI Chat Completions API**. These are different shapes for the same concept.

| What | Anthropic (CodeAgent expects) | OpenAI (JARVIS serves) |
|------|-------------------------------|------------------------|
| Endpoint | `POST /v1/messages` | `POST /v1/chat/completions` |
| System prompt | `system: [{ type: "text", text: "..." }]` (separate field, array of blocks) | `messages[0]: { role: "system", content: "..." }` (first message) |
| Assistant tool call | `{ type: "tool_use", id: "...", name: "read_file", input: {...} }` inside `content[]` | `tool_calls: [{ id: "...", function: { name: "read_file", arguments: "{...}" } }]` on the message |
| Tool result back | `{ role: "user", content: [{ type: "tool_result", tool_use_id: "...", content: "..." }] }` | `{ role: "tool", tool_call_id: "...", content: "..." }` |
| Streaming | Custom SSE events: `message_start`, `content_block_start`, `content_block_delta` (with `input_json_delta` for tool args), `content_block_stop`, `message_stop` | Standard SSE: `data: { choices: [{ delta: { tool_calls: [...] } }] }` |
| Extra fields | `betas`, `cache_control`, `thinking`, `context_management` | None of these |

**This is a translation problem.** The information is the same, just shaped differently. A ~500-line adapter can convert between them. Or JARVIS can add a native Anthropic-shaped endpoint.

### Mismatch 2: Tool Calling Capability (The Real Question)

Even after format translation, the LLM itself (Qwen3.5-27B) must be **capable** of:

1. **Receiving tool schemas** — CodeAgent sends 20+ tool definitions (read_file, edit, bash, grep, glob, etc.) as JSON schemas in every request. The model must understand these.

2. **Choosing the right tool** — Given a task like "find the bug in auth.py", the model must decide to call `read_file` first, not `bash` or `edit`.

3. **Emitting valid tool-call JSON** — The model's output must be parseable structured data, not free-form text that mentions a tool name. Example of what's needed:
   ```json
   {"name": "read_file", "arguments": {"file_path": "/src/auth.py"}}
   ```

4. **Handling the loop** — After getting a tool result back, the model must decide whether to call another tool or respond with text. A typical task involves 5-30 tool calls in sequence.

5. **Doing all this with a ~50-70K token system prompt** — CodeAgent's system prompt (instructions + all tool schemas + git status + config) is enormous. The model must still follow tool-calling conventions under this load.

**This is a capability question**, not a format question. If Qwen3.5-27B can't reliably do structured tool calling, no amount of format translation helps.

## Why JARVIS Can't Drive CodeAgent Today

JARVIS currently has **neither**:

1. **No tool-call support in the API** — `ChatCompletionRequest` doesn't accept `tools` or `tool_choice`. Responses can't carry `tool_calls`. Even if Qwen3.5 produced tool calls, the API would drop them.

2. **No tool-call support in serving** — vLLM is used in offline mode (`from vllm import LLM`), which returns raw text. The vLLM OpenAI server (which CAN parse tool calls) isn't used for real serving. And when it was used in the compat test script, the `--enable-auto-tool-choice` flag wasn't passed.

3. **No Anthropic format endpoint** — Only OpenAI-shaped `/v1/chat/completions` exists.

```
CodeAgent ──Anthropic format──▶ ???
                                 ↕ (no translation layer)
JARVIS API ──OpenAI format──▶ vLLM (offline, no tool parsing) ──▶ raw text
```

## Our Approach (Current Experiment)

We're solving this bottom-up — prove tool calling works first, then add the format translation.

### Step 1: Prove Qwen3.5 Can Do Tool Calls (in progress)

Branch: `tooluse-experiment`

```
Smoke test ──▶ Proxy (:8001) ──▶ vLLM OpenAI Server (:8290)
                passthrough        --enable-auto-tool-choice
                                   --tool-call-parser hermes
```

- Launch vLLM's OpenAI server with tool-call flags turned on
- Thin proxy forwards tool-bearing requests to vLLM
- 3 smoke tests verify: single tool call, tool-result continuation, multi-tool selection
- **Zero changes to existing JARVIS code** — all new files on a separate branch

**Status:** Submitted to Delta. vLLM + proxy came up healthy. Waiting on smoke test rerun (first run failed because pytest wasn't installed — fixed).

### Step 2: Add Anthropic Format Shim (next, if Step 1 passes)

Once we know Qwen3.5 emits valid tool calls, add a translation layer so CodeAgent can talk to JARVIS directly:

```
CodeAgent
    │
    │ Anthropic Messages format
    ▼
JARVIS Anthropic Shim (:8000)     ← NEW: translates format
    │
    │ OpenAI Chat format (with tools)
    ▼
vLLM OpenAI Server (:8290)        ← already working from Step 1
    │
    ▼
Qwen3.5-27B response (with tool_calls)
    │
    │ translated back to Anthropic SSE events
    ▼
CodeAgent (receives tool_use blocks, executes tools, continues loop)
```

CodeAgent config would just be:
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=dummy
```

Zero changes to CodeAgent itself.

### Step 3: Integration Test (future)

Run CodeAgent against JARVIS on a real task — something like "read this file and fix the typo" — and verify the full tool-call loop works end-to-end.

## Possible Outcomes of Step 1

| Result | Meaning | Next Action |
|--------|---------|-------------|
| All 3 smoke tests pass | Qwen3.5 does structured tool calls correctly | Build the Anthropic shim (Step 2) |
| Tool calls come back as plain text in `content` | Wrong vLLM parser for Qwen3.5 | Try different `--tool-call-parser` value |
| Tool call JSON is malformed | Model capability gap | Either fix via chat template, or add tool-use examples to LoRA training data |
| Model picks wrong tools / doesn't follow schemas | Model capability gap under load | May need tool-use fine-tuning or a stronger base model |

## Key Files

| File | Repo | Purpose |
|------|------|---------|
| `src/jarvis/tooluse/proxy.py` | JARVIS | Pass-through proxy to vLLM with tool support |
| `src/jarvis/tooluse/schemas.py` | JARVIS | OpenAI tool-call pydantic models |
| `src/jarvis/tooluse/server.py` | JARVIS | Standalone uvicorn entrypoint |
| `scripts/run_vllm_tooluse.sh` | JARVIS | SLURM launcher with tool-call flags |
| `tests/tooluse/test_smoke.py` | JARVIS | 3 smoke tests for tool-call validity |
| `guide/03-LLM-AND-API.md` | jarvis-body | How CodeAgent talks to LLMs |
| `guide/28-CUSTOM-MODEL-INTEGRATION.md` | jarvis-body | Custom model integration playbook |
