# JARVIS — Project Instructions

## What Is JARVIS

JARVIS is a self-hosted, routed multi-specialist AI system that serves as the reasoning backend for GRACE (a High Energy Physics research agent). It replaces expensive frontier API calls with a locally-deployed ensemble of specialist models behind an OpenAI-compatible API.

**JARVIS is NOT a single model or a Mixture of Experts.** It is a system — a query-level router dispatching to independent specialist brains, wrapped in an inference amplification layer, served via a unified API. Each brain can be swapped, upgraded, or extended independently.

## Architecture Overview

```
GRACE (HEP Agent) ──HTTP──▶ JARVIS API (localhost:8000)
                                    │
                              ┌─────┴─────┐
                              │   ROUTER   │  (domain + difficulty classifier)
                              └──┬──┬──┬──┘
                                 │  │  │
                    ┌────────────┘  │  └────────────┐
                    ▼              ▼                ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
              │   MATH   │ │ PHYSICS  │ │   CODE   │ │ SPECIALISTS  │
              │  70B/32B │ │   32B    │ │   32B    │ │  7B models   │
              │ off-shelf│ │ custom   │ │ custom   │ │ on-demand    │
              └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
                   └────────┬───┘────────────┘───────────────┘
                            ▼
                   ┌─────────────────┐
                   │ INFERENCE ENGINE│  (best-of-N, verification, budget forcing)
                   └────────┬────────┘
                            ▼
                      JSON response
```

## Key Technical Decisions

- **Deployment target:** NVIDIA DGX Spark (128GB unified RAM, GB10 Blackwell, $4,699)
- **Physics brain base:** R1-Distill-Qwen-32B (Qwen2.5 architecture, 62.1% GPQA baseline, R1 reasoning distillation)
- **Code brain base:** Qwen2.5-Coder-32B-Instruct (Qwen2.5 architecture, purpose-built for code, HumanEval 88.4%, LiveCodeBench ~40-50%)
- **Two separate bases (~32 GB total at FP4).** Both now use Qwen2.5 architecture. LoRA adapters are still base-specific (trained on different data).
- **Math brain:** R1-Distill-Llama-70B (off-shelf, no training) or R1-Distill-Qwen-32B with math LoRA (if memory-constrained)
- **Router:** Lightweight BERT classifier (~110M params), two-stage: domain → difficulty
- **Specialist models:** Standalone 7B models (ChemLLM, BioMistral, ESM3-open, Evo 2) loaded from SSD on demand
- **API format:** OpenAI-compatible `/v1/chat/completions` endpoint
- **Inference framework:** vLLM or TensorRT-LLM on DGX Spark (CUDA/Blackwell native)
- **Quantization:** NVFP4 (native Blackwell support, <1% accuracy loss)

## Project Structure

```
jarvis/
├── claude.md                  # This file — project context for Claude
├── docs/
│   ├── ARCHITECTURE.md        # Component design, interfaces, data flow
│   ├── ROADMAP.md             # Phased development plan with milestones
│   ├── TRAINING_PIPELINE.md   # Training instructions for ACCESS-CI Delta
│   ├── DEPLOYMENT.md          # DGX Spark setup and deployment guide
│   ├── MODELS.md              # Registry of all models with specs
│   └── API_SPEC.md            # OpenAI-compatible API specification
├── src/
│   ├── router/                # Domain + difficulty classification
│   ├── brains/                # Brain management, LoRA loading, model configs
│   ├── inference/             # Amplification engine (best-of-N, verification, etc.)
│   │                          # Includes context_manager.py for KV cache optimization
│   ├── specialists/           # Specialist model registry and on-demand loading
│   ├── api/                   # FastAPI server, OpenAI-compatible endpoints
│   └── rag/                   # FAISS knowledge base for physics
├── training/
│   ├── physics/               # Physics brain training scripts (Delta)
│   ├── code/                  # Code brain training scripts (Delta)
│   ├── router/                # Router classifier training
│   └── data/                  # Data generation and curation pipelines
├── configs/
│   ├── models.yaml            # Model registry and download paths
│   ├── router.yaml            # Router thresholds and domain mappings
│   ├── inference.yaml         # Inference engine settings per difficulty level
│   └── deployment.yaml        # Hardware-specific deployment configs
├── tests/
│   ├── benchmarks/            # GPQA, AIME, LiveCodeBench evaluation harnesses
│   └── integration/           # End-to-end API tests
└── scripts/
    ├── download_models.sh     # Download and verify all model weights
    ├── setup_dgx_spark.sh     # DGX Spark environment setup
    └── benchmark.sh           # Run full evaluation suite
```

## HPC / SLURM Rules

- **SLURM account:** Always use `--account=bgde-delta-gpu` for ALL JARVIS jobs on Delta. No exceptions. This is the ONLY allocation for this project.
- **Python module:** `module load python/3.13.5-gcc13.3.1` (NOT anaconda3, which doesn't exist on Delta). System Python is 3.9 — too old.
- **Environments:** Use plain Python venv, not conda.

## Git & Commit Rules

- **No AI co-authorship lines.** Never add `Co-Authored-By` or any similar attribution to Claude, Anthropic, or any AI in commits.
- **No AI signatures or branding** in commit messages, PR descriptions, code comments, or any generated content.

## Development Principles

1. **Modular everything.** Each brain, the router, the inference engine, and each specialist are independent components with clean interfaces. Adding a new specialist should require zero changes to existing code.

2. **Configuration-driven.** Model paths, LoRA adapters, inference settings, routing thresholds — all in YAML configs, not hardcoded. Swapping a brain or adding a domain is a config change.

3. **OpenAI-compatible API first.** GRACE (and any future client) talks to JARVIS through the standard OpenAI chat completions format. Internal routing is invisible to clients.

4. **Test against real benchmarks.** Every change must be evaluated against GPQA Diamond (physics), LiveCodeBench (code), and AIME 2024 (math). No vibes-based assessment.

5. **Memory-aware.** Every model loading decision must account for the 128GB RAM budget. The system should track active memory usage and swap models intelligently.

## Current Status

**Phases 0-6 complete (0-3 validated on Delta).** Full serving stack: vLLM inference, 8-domain router, difficulty-aware amplification (single_pass/best-of-N/verified), specialist loading with LRU eviction (ESM3/Evo2 adapters), RAG for physics queries (30-passage corpus). 142 tests passing. Code brain swapped from Qwen3-32B to Qwen2.5-Coder-32B-Instruct (purpose-built, +20% baseline). S* code execution verification planned. Remaining: Phase 4 (Training on Delta — 8,000 SUs, code budget reallocated to physics GRPO), Phase 7 (GRACE Integration), Phase 8 (Optimization).

## Key Reference Documents

- `docs/ARCHITECTURE.md` — How the components connect, data flow, interface contracts
- `docs/ROADMAP.md` — What to build, in what order, with milestones
- `docs/TRAINING_PIPELINE.md` — How to train the physics and code brains on Delta
- `docs/DEPLOYMENT.md` — How to deploy on DGX Spark
- `docs/MODELS.md` — Every model we use, its size, source, and purpose
- `docs/API_SPEC.md` — The API contract between GRACE and JARVIS
