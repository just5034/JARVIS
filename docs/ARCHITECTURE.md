# JARVIS Architecture

## System Overview

JARVIS is a difficulty-aware inference system with specialist routing. It is composed of five major subsystems:

1. **API Server** — Receives requests, returns responses. OpenAI-compatible.
2. **Router** — Classifies queries by domain and difficulty.
3. **Brain Manager** — Loads and manages the base model, LoRA adapters, and specialists.
4. **Inference Engine** — Applies difficulty-aware amplification (best-of-N, verification, budget forcing).
5. **Specialist Registry** — Manages on-demand loading of domain specialist models from disk.

## Request Lifecycle

```
1. GRACE (or any client) sends POST /v1/chat/completions with a query
2. API Server receives the request
3. Router classifies: { domain: "physics", difficulty: "hard", hep: true }
4. Brain Manager ensures correct configuration:
   - Base model (Qwen3.5-27B) is always resident
   - If HEP detected → hot-swap HEP LoRA adapter (milliseconds)
   - If specialist domain (chemistry, protein, etc.) → load specialist from SSD
5. Inference Engine selects strategy based on difficulty:
   - Easy → single forward pass
   - Medium → best-of-4 + self-consistency voting
   - Hard → best-of-16 + ThinkPRM verification + budget forcing + verification chain
6. RAG module (if physics) retrieves relevant knowledge passages, prepends to prompt
7. Model generates response(s)
8. Inference Engine selects/verifies best response
9. API Server returns the response in OpenAI format
```

## Component Specifications

### 1. API Server (`src/api/`)

**Framework:** FastAPI

**Endpoints:**
- `POST /v1/chat/completions` — Main inference endpoint (OpenAI-compatible)
- `POST /v1/completions` — Legacy completions format
- `GET /v1/models` — List available models/brains
- `GET /health` — System health, memory usage, loaded models
- `POST /admin/load` — Manually load/unload a model or adapter
- `GET /admin/memory` — Current RAM usage breakdown

**Behavior:**
- Streaming support via SSE (Server-Sent Events)
- Request timeout configurable per difficulty level (easy: 30s, medium: 120s, hard: 600s)
- The `model` field in the request is optional — if omitted, the router decides. If specified (e.g., `model: "chemistry"`), routing is overridden to that specialist.

### 2. Router (`src/router/`)

**Architecture:** Two lightweight BERT classifiers running sequentially, plus keyword-based HEP detection.

**Stage 1 — Domain Classification:**
- Input: query text (last user message + system prompt if present)
- Output: one of `math`, `physics`, `code`, `chemistry`, `biology`, `genomics`, `protein`, `general`
- Purpose: Determines RAG activation (physics), specialist dispatch (chemistry/biology/protein/genomics), HEP LoRA activation (physics/code + HEP keywords), and code verification strategy (code)
- Model: `bert-base-uncased` fine-tuned, ~110M params (~0.06 GB)

**Stage 2 — Difficulty Estimation:**
- Input: query text + predicted domain
- Output: one of `easy`, `medium`, `hard`
- Purpose: Drives inference strategy selection (single-pass vs best-of-N vs verified)
- Model: separate `bert-base-uncased` fine-tuned

**Stage 3 — HEP Subdomain Detection (optional):**
- If domain = physics or code, detect if query is HEP-specific
- If HEP detected → signal Brain Manager to hot-swap HEP LoRA adapter
- Keywords + classifier hybrid approach

**Extensibility:** Adding a new specialist domain requires:
1. Add domain label to Stage 1 classifier
2. Provide ~1K-5K labeled training examples
3. Retrain classifier (~minutes on CPU)
4. Register the new specialist in `configs/models.yaml`

### 3. Brain Manager (`src/brains/`)

**Responsibility:** Manages base model loading, LoRA adapter swapping, and memory tracking.

**Memory Model:**
```yaml
# Always resident (core system ~21 GB)
permanent:
  - qwen35_27b: 14.0 GB          # Single unified base model (FP4)
  - router: 0.06 GB
  - think_prm: 0.8 GB            # ThinkPRM 1.5B verifier
  - draft_model: 0.4 GB          # Speculative decoding draft (Qwen3.5-0.8B)
  - rag_index: 5.0 GB            # FAISS physics knowledge base
  - framework_overhead: 7.0 GB   # vLLM + OS
  - total: ~27.66 GB

# Swappable LoRA adapters (hot-swap in milliseconds)
lora_adapters:
  - hep_physics: 0.3 GB          # Particle physics, detector design
  - hep_code: 0.3 GB             # Geant4, ROOT, Pythia8 patterns

# On-demand specialists (loaded from SSD, 5-10 seconds)
specialists:
  - esm3_open: 0.7 GB
  - evo2_7b: 3.5 GB
  - chemllm_7b: 3.5 GB
  - biomistral_7b: 3.5 GB
```

**Single-base architecture rationale:** Qwen3.5-27B (released Feb 2026) surpasses our original targets across all benchmarks — GPQA Diamond 86%, LiveCodeBench 80.7%, AIME 81%. A single 14 GB model (FP4) replaces two 16 GB models (R1-Distill-Qwen-32B + Qwen2.5-Coder-32B-Instruct), freeing ~18 GB for longer context windows, more parallel candidates, and more simultaneously-loaded specialists. Domain-specific expertise is provided by HEP LoRA adapters, not separate base models.

**Adapter Swapping Logic:**
1. Router determines domain → checks if HEP-specific
2. If HEP detected and correct adapter not active → swap LoRA adapter (milliseconds)
3. If non-HEP query and adapter is active → unload adapter (milliseconds)
4. If specialist domain (chemistry, protein, etc.) → load specialist from SSD if not resident, with LRU eviction if memory tight

**Memory Tracking:** Maintain a real-time ledger of loaded models and their memory footprints. Refuse to load a model if it would exceed the 128GB budget minus a 5GB safety margin. With ~28 GB core, approximately 95 GB remains for specialists and KV cache.

### 4. Inference Engine (`src/inference/`)

**Difficulty-aware strategy selection:**

| Difficulty | Strategy | Approx Cost | Timeout |
|-----------|----------|-------------|---------|
| Easy | Single pass, speculative decoding | 1x | 30s |
| Medium | Best-of-4, self-consistency voting | 4x | 120s |
| Hard | Best-of-16, ThinkPRM verification, budget forcing, verification chain | 16-32x | 600s |

**Subcomponents:**

**a) Speculative Decoding** — Applied to all queries. Uses a small Qwen3.5-compatible draft model to propose tokens, main model verifies. 2-3x throughput improvement.

**b) Self-Consistency Voting** — Generate N responses, extract final answers, select by majority vote. Used for medium+ difficulty.

**c) ThinkPRM Verification** — Off-the-shelf 1.5B generative process reward model. Scores each candidate solution's reasoning chain. Used for hard queries with pessimistic selection (select solution with least verification uncertainty).

**d) Budget Forcing ("Wait" Trick)** — For hard queries, if the model produces a conclusion token before reaching the thinking budget, append "Wait" to force continued self-examination. Up to 3 appends. Implemented at the inference server level.

**e) Verification Chains** — Append "Verify your answer by [substituting back / checking dimensional analysis / testing edge cases]. If you find an error, correct it." to medium/hard prompts.

**f) S\* Code Verification** — For code domain hard queries. Generate candidates → generate distinguishing test inputs → execute all candidates → select by correct behavior. Requires sandboxed Python executor.

**g) Context Window & KV Cache Management:**

The context window is constrained by both model architecture and available RAM for KV cache storage.

**Model architecture limits:**

| Model | Architecture Context | Recommended Max |
|-------|---------------------|----------------|
| Qwen3.5-27B | 262K native (1M extended) | 131K |
| 7B specialists | Varies (most 32K-128K) | 32K |

**KV cache memory cost (27B model, FP8 KV):**

| Context Length | FP8 KV | 2-bit KV (KVQuant) |
|---------------|--------|-------------------|
| 32K | ~3 GB | ~0.75 GB |
| 64K | ~6 GB | ~1.5 GB |
| 128K | ~12 GB | ~3 GB |

**Practical context limits on DGX Spark (128GB RAM, ~95GB available after core system):**

| Difficulty | Sampling | KV Cache Config | Max Context | KV Memory |
|-----------|----------|----------------|-------------|-----------|
| Easy | 1 pass | FP8 | 131K | ~12 GB |
| Medium | best-of-4 parallel | FP8 | 64K | ~24 GB (4 x 6 GB) |
| Hard | best-of-16 parallel | FP8 | 32K | ~48 GB (16 x 3 GB) |
| Hard | best-of-16 parallel | 2-bit KV | 64K | ~24 GB (16 x 1.5 GB) |
| Hard | best-of-16 parallel | 2-bit KV | 128K | ~48 GB (16 x 3 GB) |
| Derivation | 1 pass + SSD offload | FP8 + offload | 131K+ | ~5 GB active + SSD |

**Default configuration:** FP8 KV cache enabled globally. 2-bit KVQuant enabled for hard queries. SSD offload available for physics derivation mode.

**h) Extended Thinking Budgets:**
- Easy: 4K tokens
- Medium: 16K tokens
- Hard: 32-64K tokens
- Physics derivations: up to 128K (with FP8 KV + SSD offload)

### 5. Specialist Registry (`src/specialists/`)

**Responsibility:** Maintains catalog of available specialist models, handles on-demand loading/eviction.

**Registry format** (in `configs/models.yaml`):
```yaml
specialists:
  chemistry:
    model_id: "AI4Chem/ChemLLM-7B-Chat"
    path: "/models/specialists/chemllm-7b/"
    size_gb: 3.5
    quantization: "fp4"
    type: "text_llm"           # vs "protein_model", "dna_model"
    router_domain: "chemistry"
    load_priority: "on_demand"  # vs "always_resident"

  proteins:
    model_id: "EvolutionaryScale/esm3-open"
    path: "/models/specialists/esm3-open/"
    size_gb: 0.7
    quantization: "fp16"        # Small enough to run at full precision
    type: "protein_model"
    router_domain: "protein"
    load_priority: "on_demand"
    api_adapter: "esm3"         # Needs custom input/output handling
```

**Non-text specialists** (ESM3, Evo 2) require custom API adapters that translate between the OpenAI chat format and the model's native input format (protein sequences, DNA strings). These adapters live in `src/specialists/adapters/`.

**Plug-and-play extensibility:** Adding a new specialist requires only a config entry and (for non-text models) an adapter. The base model can also be swapped by updating `configs/models.yaml` and retraining any LoRA adapters — the inference engine, API, router, and specialists are all model-agnostic.

### 6. RAG Module (`src/rag/`)

**Purpose:** Augments physics domain queries with retrieved knowledge passages.

**Implementation:**
- FAISS vector index (~5GB) over physics/chemistry/biology reference corpus
- Embedding model: `all-MiniLM-L6-v2` (22M params, negligible memory)
- At inference time: embed query → retrieve top-5 passages → prepend to prompt
- Corpus: physical constants, key equations, derivations, reaction mechanisms, unit conversions

**Only activated for physics-domain queries.** Other domains bypass RAG.

## Data Flow Diagram

```
┌─────────┐    ┌──────────┐    ┌────────────┐    ┌───────────────┐
│  Client  │───▶│ API      │───▶│  Router    │───▶│ Brain Manager │
│ (GRACE)  │    │ Server   │    │ (classify) │    │ (LoRA/spec.)  │
└─────────┘    └────┬─────┘    └────────────┘    └───────┬───────┘
                    │                                     │
                    │           ┌────────────┐            │
                    │           │  RAG       │◀───────────┤ (if physics)
                    │           │  (retrieve) │            │
                    │           └─────┬──────┘            │
                    │                 │                    │
                    │           ┌─────▼──────┐    ┌───────▼───────┐
                    │           │  Inference  │◀──│  Qwen3.5-27B  │
                    │           │  Engine     │   │  + optional   │
                    │           │ (amplify)   │   │  HEP LoRA     │
                    │           └─────┬──────┘   └───────────────┘
                    │                 │
                    ◀─────────────────┘
               JSON response
```

## Concurrency Model

- The API server handles multiple concurrent requests via FastAPI's async support
- vLLM handles batching internally — multiple queries to the same model are batched for GPU efficiency
- LoRA adapter swaps are serialized — only one swap can happen at a time (protected by async lock)
- The router runs on CPU and is always available regardless of GPU state
- Specialist loading is async — the API returns a "processing" status if a model needs to be loaded from SSD

## Error Handling

- If a model fails to load (OOM, corrupted weights): return 503 with specific error
- If inference times out: return partial response if streaming, 504 if not
- If router confidence is below threshold: use base Qwen3.5-27B without any LoRA (general-purpose mode)
- If a specialist adapter is missing: fall back to Qwen3.5-27B base model (strong general capabilities cover most domains)

## Future-Proofing

The architecture is designed for plug-and-play model upgrades:
- **Base model swap:** Update `configs/models.yaml`, retrain LoRA adapters, verify vLLM support. All other components (API, router, inference engine, specialists) are model-agnostic.
- **New specialist:** Add config entry + optional adapter. Zero code changes to existing components.
- **New LoRA domain:** Train adapter, add to config, add keywords to router. Existing adapters unaffected.
