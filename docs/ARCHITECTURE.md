# JARVIS Architecture

## System Overview

JARVIS is a routed ensemble inference system. It is composed of five major subsystems:

1. **API Server** — Receives requests, returns responses. OpenAI-compatible.
2. **Router** — Classifies queries by domain and difficulty.
3. **Brain Manager** — Loads, swaps, and manages model weights and LoRA adapters.
4. **Inference Engine** — Applies difficulty-aware amplification (best-of-N, verification, budget forcing).
5. **Specialist Registry** — Manages on-demand loading of domain specialist models from disk.

## Request Lifecycle

```
1. GRACE sends POST /v1/chat/completions with a query
2. API Server receives the request
3. Router classifies: { domain: "physics", difficulty: "hard", subdomain: "qft" }
4. Brain Manager ensures the correct model + LoRA adapter is loaded
   - If physics brain with HEP adapter is already resident → proceed
   - If not → swap LoRA adapter (milliseconds) or load model from SSD (seconds)
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
- The `model` field in the request is optional — if omitted, the router decides. If specified (e.g., `model: "physics"` or `model: "chemistry"`), routing is overridden.

### 2. Router (`src/router/`)

**Architecture:** Two lightweight BERT classifiers running sequentially.

**Stage 1 — Domain Classification:**
- Input: query text (last user message + system prompt if present)
- Output: one of `math`, `physics`, `code`, `chemistry`, `biology`, `genomics`, `protein`, `general`, or any registered specialist domain
- Training data: ~5K labeled examples per core domain, expandable
- Model: `bert-base-uncased` fine-tuned, ~110M params (~0.06 GB)

**Stage 2 — Difficulty Estimation:**
- Input: query text + predicted domain
- Output: one of `easy`, `medium`, `hard`
- Training data: generated automatically by running each brain on a validation set
  - Correct in 1 pass = easy
  - Correct in best-of-4 = medium
  - Incorrect or only correct in best-of-16 = hard
- Model: separate `bert-base-uncased` fine-tuned

**Stage 3 — HEP Subdomain Detection (optional):**
- If domain = physics or code, detect if query is HEP-specific
- If HEP detected → signal Brain Manager to hot-swap HEP LoRA adapter
- Keywords + classifier hybrid approach

**Extensibility:** Adding a new domain requires:
1. Add domain label to Stage 1 classifier
2. Provide ~1K-5K labeled training examples
3. Retrain classifier (~minutes on CPU)
4. Register the new brain/specialist in `configs/models.yaml`

### 3. Brain Manager (`src/brains/`)

**Responsibility:** Manages model loading, LoRA adapter swapping, and memory tracking.

**Memory Model:**
```yaml
# Always resident (core system ~34 GB at FP4)
permanent:
  - router: 0.06 GB
  - think_prm: 0.8 GB          # ThinkPRM 1.5B verifier
  - draft_model: 0.8 GB         # Speculative decoding draft (1.5B)
  - rag_index: 5.0 GB           # FAISS physics knowledge base
  - framework_overhead: 10.0 GB  # vLLM/TensorRT-LLM + OS

# Swappable (loaded based on routing decisions)
swappable:
  base_models:
    - qwen_32b: 16.0 GB         # Shared base for physics/code/math adapters
    - r1_distill_70b: 35.0 GB   # Math brain (if using 70B config)
  lora_adapters:                  # Hot-swap in milliseconds
    - physics_general: 0.3 GB
    - physics_hep: 0.3 GB
    - code_general: 0.3 GB
    - code_hep: 0.3 GB
    - math_adapter: 0.3 GB       # Only if using 32B math config

# On-demand specialists (loaded from SSD, 5-10 seconds)
specialists:
  - esm3_open: 0.7 GB
  - evo2_7b: 3.5 GB
  - chemllm_7b: 3.5 GB
  - biomistral_7b: 3.5 GB
```

**Adapter Swapping Logic:**
1. Check if requested domain's adapter is already active → proceed immediately
2. If different adapter needed on same base model → unload current adapter, load new one (milliseconds)
3. If different base model needed → check memory budget → unload old base if necessary → load new base from SSD
4. If specialist model needed → check if already resident → if not, load from SSD, optionally evicting least-recently-used specialist

**Memory Tracking:** Maintain a real-time ledger of loaded models and their memory footprints. Refuse to load a model if it would exceed the 128GB budget minus a 5GB safety margin.

### 4. Inference Engine (`src/inference/`)

**Difficulty-aware strategy selection:**

| Difficulty | Strategy | Approx Cost | Timeout |
|-----------|----------|-------------|---------|
| Easy | Single pass, speculative decoding | 1× | 30s |
| Medium | Best-of-4, self-consistency voting | 4× | 120s |
| Hard | Best-of-16, ThinkPRM verification, budget forcing, verification chain | 16-32× | 600s |

**Subcomponents:**

**a) Speculative Decoding** — Applied to all queries. Uses R1-Distill-Qwen-1.5B as draft model to propose tokens, main model verifies. 2-3× throughput improvement.

**b) Self-Consistency Voting** — Generate N responses, extract final answers, select by majority vote. Used for medium+ difficulty.

**c) ThinkPRM Verification** — Off-the-shelf 1.5B generative process reward model. Scores each candidate solution's reasoning chain. Used for hard queries with pessimistic selection (select solution with least verification uncertainty).

**d) Budget Forcing ("Wait" Trick)** — For hard queries, if the model produces a conclusion token before reaching the thinking budget, append "Wait" to force continued self-examination. Up to 3 appends. Implemented at the inference server level.

**e) Verification Chains** — Append "Verify your answer by [substituting back / checking dimensional analysis / testing edge cases]. If you find an error, correct it." to medium/hard prompts.

**f) S\* Code Verification** — For code brain hard queries only. Generate 16 candidates → generate distinguishing test inputs → execute all candidates → select by correct behavior. Requires sandboxed Python executor.

**g) Extended Thinking Budgets:**
- Easy: 4K tokens
- Medium: 16K tokens
- Hard: 32-64K tokens
- Physics derivations: up to 128K (full context window)

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

### 6. RAG Module (`src/rag/`)

**Purpose:** Augments physics brain queries with retrieved knowledge passages.

**Implementation:**
- FAISS vector index (~5GB) over physics/chemistry/biology reference corpus
- Embedding model: `all-MiniLM-L6-v2` (22M params, negligible memory)
- At inference time: embed query → retrieve top-5 passages → prepend to prompt
- Corpus: physical constants, key equations, derivations, reaction mechanisms, unit conversions

**Only activated for physics-domain queries.** Other domains bypass RAG.

## Data Flow Diagram

```
┌─────────┐    ┌──────────┐    ┌────────────┐    ┌───────────────┐
│  GRACE   │───▶│ API      │───▶│  Router    │───▶│ Brain Manager │
│ (client) │    │ Server   │    │ (classify) │    │ (load/swap)   │
└─────────┘    └────┬─────┘    └────────────┘    └───────┬───────┘
                    │                                     │
                    │           ┌────────────┐            │
                    │           │  RAG       │◀───────────┤ (if physics)
                    │           │  (retrieve) │            │
                    │           └─────┬──────┘            │
                    │                 │                    │
                    │           ┌─────▼──────┐    ┌───────▼───────┐
                    │           │  Inference  │◀──│  Active Model │
                    │           │  Engine     │   │  (GPU memory) │
                    │           │ (amplify)   │   └───────────────┘
                    │           └─────┬──────┘
                    │                 │
                    ◀─────────────────┘
               JSON response
```

## Concurrency Model

- The API server handles multiple concurrent requests via FastAPI's async support
- vLLM/TensorRT-LLM handles batching internally — multiple queries to the same model are batched for GPU efficiency
- Model swaps are serialized — only one swap can happen at a time (protected by async lock)
- The router runs on CPU and is always available regardless of GPU state
- Specialist loading is async — the API returns a "processing" status if a model needs to be loaded from SSD

## Error Handling

- If a model fails to load (OOM, corrupted weights): return 503 with specific error
- If inference times out: return partial response if streaming, 504 if not
- If router confidence is below threshold: fall back to the Qwen-32B base model without any LoRA (general-purpose mode)
- If a specialist adapter is missing: fall back to the closest available brain (e.g., chemistry query with no ChemLLM → physics brain)
