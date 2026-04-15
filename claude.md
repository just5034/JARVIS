# JARVIS — Project Instructions

## What Is JARVIS

JARVIS is a self-hosted, difficulty-aware inference system with specialist routing that serves as the reasoning backend for GRACE (a High Energy Physics research agent) and as a general-purpose LLM replacement for frontier API calls. It deploys a single strong base model behind an OpenAI-compatible API, augmented by inference amplification (best-of-N, verification, budget forcing), domain-specific LoRA adapters, and on-demand specialist models.

**JARVIS is a system, not a single model.** It wraps a unified base model (Qwen3.5-27B) in a difficulty-aware inference layer, with specialist routing for non-LLM domains (proteins, genomics) and HEP-specific LoRA adapters for domain expertise. The base model, adapters, and specialists can all be swapped independently.

## Architecture Overview

```
Client (GRACE / any app) ──HTTP──▶ JARVIS API (localhost:8000)
                                           │
                                     ┌─────┴─────┐
                                     │  ROUTER   │  (difficulty + domain classifier)
                                     └──┬──┬──┬──┘
                                        │  │  │
                           ┌────────────┘  │  └────────────┐
                           ▼               ▼               ▼
                    ┌────────────┐  ┌────────────┐  ┌──────────────┐
                    │ Qwen3.5    │  │ Qwen3.5    │  │ SPECIALISTS  │
                    │  27B       │  │  27B       │  │  7B models   │
                    │ easy →     │  │ hard →     │  │  on-demand   │
                    │ single pass│  │ best-of-N  │  │              │
                    └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
                          └───────┬───────┘────────────────┘
                                  ▼
                         ┌─────────────────┐
                         │ INFERENCE ENGINE │  (verification, budget forcing)
                         └────────┬────────┘
                                  ▼
                            JSON response
```

## Key Technical Decisions

- **Deployment target:** NVIDIA DGX Spark (128GB unified RAM, GB10 Blackwell, $4,699)
- **Unified base model:** Qwen3.5-27B (dense, 27B params, ~14 GB at FP4). Handles physics, math, code, and general queries. GPQA Diamond 86%, LiveCodeBench 80.7%, AIME 81%.
- **HEP adapters:** Two LoRA adapters (hep_physics, hep_code) hot-swapped when HEP content detected.
- **No separate math/physics/code brains.** Qwen3.5-27B exceeds all original per-domain targets. Domain classification still used for specialist dispatch, RAG activation, and HEP LoRA triggers.
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
│   ├── physics/               # HEP physics LoRA training scripts (Delta)
│   ├── code/                  # HEP code LoRA training scripts (Delta)
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

### Delta storage paths (as of 2026-04-14 — `/scratch` is broken)

NCSA's Lustre quota for `/scratch/bgde` is currently corrupted: reports ~481 GB used against a 500 GB group cap even though actual content is <120 MB. Orphaned OST objects. Ticket open with `help@ncsa.illinois.edu` (filed 2026-04-14). Until resolved, **do not write new large data to `/scratch`** — it counts against the bogus phantom quota and will fail.

**Route all new writes to:**
- `/work/hdd/bgde/jhill5/jarvis-venv` — Python venv
- `/work/hdd/bgde/jhill5/hf_cache` — `HF_HOME`
- `/work/hdd/bgde/jhill5/logs` — SLURM/vLLM logs
- `$TMPDIR` — ephemeral workspaces (SWE-bench clones already use this)
- `/projects/bgde/jhill5/models/` — persistent model weights (untouched by the bug)

`/scratch/bgde/jhill5/eval`, `data`, `tb_logs` remain with preserved research artifacts (~116 MB total, also backed up to `/u/jhill5/scratch_backup/`). Reading is fine; writing there is not, until NCSA resolves.

When submitting SLURM scripts, `export HF_HOME=/work/hdd/bgde/jhill5/hf_cache` and `source /work/hdd/bgde/jhill5/jarvis-venv/bin/activate` instead of the old `/scratch` paths.

## Git & Commit Rules

- **No AI co-authorship lines.** Never add `Co-Authored-By` or any similar attribution to Claude, Anthropic, or any AI in commits.
- **No AI signatures or branding** in commit messages, PR descriptions, code comments, or any generated content.

## Development Principles

1. **Modular everything.** The base model, router, inference engine, and each specialist are independent components with clean interfaces. Adding a new specialist should require zero changes to existing code. Swapping the base model is a config change + LoRA retrain.

2. **Configuration-driven.** Model paths, LoRA adapters, inference settings, routing thresholds — all in YAML configs, not hardcoded. Swapping a brain or adding a domain is a config change.

3. **OpenAI-compatible API first.** GRACE (and any future client) talks to JARVIS through the standard OpenAI chat completions format. Internal routing is invisible to clients.

4. **Test against real benchmarks.** Every change must be evaluated against GPQA Diamond (physics), LiveCodeBench (code), and AIME 2024 (math). No vibes-based assessment.

5. **Memory-aware.** Every model loading decision must account for the 128GB RAM budget. The system should track active memory usage and swap models intelligently.

## Current Status

**Phases 0-6 complete (0-3 validated on Delta).** Full serving stack: vLLM inference, 8-domain router, difficulty-aware amplification (single_pass/best-of-N/verified), specialist loading with LRU eviction (ESM3/Evo2 adapters), RAG for physics queries (30-passage corpus). 142 tests passing. S* code execution verification implemented.

**Migration in progress (2026-04-01):** Pivoting from dual-base (R1-Distill-Qwen-32B + Qwen2.5-Coder-32B-Instruct) to single Qwen3.5-27B. Configs and docs updated. Next: cancel old SFT job, download Qwen3.5-27B to Delta, run baseline evals, simplify brain_manager.py and router code, verify inference pipeline compatibility (ThinkPRM, budget forcing), train HEP LoRA adapters.

**Phase 4A trace generation (old model) completed** — 5,000 traces archived. Phase 4B SFT job (17177608) pending cancellation — was training adapter for deprecated R1-Distill-Qwen-32B.

**Budget:** ~8,000 SUs total, ~76 SU spent. ~7,924 remaining for baseline evals, HEP LoRA training, and optional GRPO.

## Key Reference Documents

- `docs/ARCHITECTURE.md` — How the components connect, data flow, interface contracts
- `docs/ROADMAP.md` — What to build, in what order, with milestones
- `docs/TRAINING_PIPELINE.md` — How to train HEP LoRA adapters on Delta
- `docs/DEPLOYMENT.md` — How to deploy on DGX Spark
- `docs/MODELS.md` — Every model we use, its size, source, and purpose
- `docs/API_SPEC.md` — The API contract between GRACE and JARVIS
