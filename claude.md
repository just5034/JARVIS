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

### Delta storage paths

`/scratch` is a bind mount to `/work/hdd`. `lfs quota` sums `/work/hdd` + `/work/nvme` (they share MDTs on the same Lustre FS). NCSA confirmed 2026-04-15 — not a bug.

**Write paths:**
- `/work/hdd/bgde/jhill5/jarvis-venv` — Python venv
- `/work/hdd/bgde/jhill5/hf_cache` — `HF_HOME`
- `/work/hdd/bgde/jhill5/logs` — SLURM/vLLM logs
- `/work/hdd/bgde/jhill5/data` — training data, traces, filtered JSONL
- `/work/hdd/bgde/jhill5/checkpoints` — training checkpoints
- `/work/hdd/bgde/jhill5/eval` — eval outputs
- `$TMPDIR` — ephemeral workspaces (SWE-bench clones, node-local SSD)
- `/projects/bgde/jhill5/models/` — persistent model weights
- `/projects/bgde/jhill5/adapters/` — final LoRA adapters

All scripts under `scripts/` and `training/` have been updated to use `/work/hdd/bgde/jhill5/` (commit 6e6afaf, 2026-04-22).

When submitting SLURM scripts, `export HF_HOME=/work/hdd/bgde/jhill5/hf_cache` and `source /work/hdd/bgde/jhill5/jarvis-venv/bin/activate`.

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

**Phases 0-6 complete (0-3 validated on Delta).** Full serving stack: vLLM inference, 8-domain router, difficulty-aware amplification (single_pass/best-of-N/verified), specialist loading with LRU eviction (ESM3/Evo2 adapters), RAG for physics queries (30-passage corpus). 164 tests passing. S* code execution verification implemented.

**Qwen3.5-27B migration complete (2026-04-22).** Single-base architecture. Baselines validated on Delta: GPQA 85.4%, AIME 89.2%, LiveCodeBench 82.5%. SWE-bench validated at 50% on astropy-heavy sample (consistent with published 72.4%) using mini-swe-agent framework. Inference amplification (budget forcing, voting, ThinkPRM) updated for Qwen3.5's "Thinking Process:" format (not `<think>` tags).

**HEP LoRA pipeline — Phases 0-3 drafted locally (2026-04-28).** Phase 0 (path/format fixes) committed earlier. Phase 1 created `training/data/extract_hep_{physics,code}.py` — both run end-to-end against the GRACE repo and emit JSONL problem sets. Phase 2 added `training/data/build_code_problems.py` and updated `build_physics_problems.py` to ingest the GRACE-extracted problems. Phase 3 added a `--domain code` flag to `training/physics/rejection_sample.py` (Python AST / GDML XML / C++ bracket-balance validation in lieu of physics-style answer extraction) and created `scripts/run_code_trace_generation.sh` (mirrors the physics SLURM script with port 8193 to allow concurrent runs). **Next: Phase 4 — first SU-spending phase.** On Delta: run extract → build → submit both `run_trace_generation.sh` (physics) and `run_code_trace_generation.sh` (code) in parallel. After both land, `sbatch scripts/run_hep_sft.sh --physics` then `--code`.

**DeepSeek V4 archived to D:\jarvis-models\ (2026-04-28).** V4 released 2026-04-24 (MIT-licensed, 1M context, native FP4+FP8). Neither variant fits DGX Spark's 128 GB RAM (V4-Flash = 150 GB, V4-Pro = 850 GB). Archived to a 2 TB external SSD via `scripts/archive_v4_to_ssd.ps1` so weights persist if HF removes them. Registered in `configs/models.yaml` under `deferred_backends:` with `load_policy: archived`; flipping that to `on_demand` activates them when hardware lands. Strategy: stay on Qwen3.5-27B; revisit hardware when (a) DeepSeek ships a V4 distill in the 27-40B range or (b) a Mac Studio M-Ultra 256GB is committed to as a hard-query escalation node. **No hardware purchase planned for now.**

**Budget:** ~8,000 SUs total, ~100 SU spent. ~7,900 remaining for HEP LoRA training (~470 SU planned, first ~50 SU triggers when Phase 4 trace generation submits).

## Key Reference Documents

- `docs/ARCHITECTURE.md` — How the components connect, data flow, interface contracts
- `docs/ROADMAP.md` — What to build, in what order, with milestones
- `docs/TRAINING_PIPELINE.md` — How to train HEP LoRA adapters on Delta
- `docs/DEPLOYMENT.md` — How to deploy on DGX Spark
- `docs/MODELS.md` — Every model we use, its size, source, and purpose
- `docs/API_SPEC.md` — The API contract between GRACE and JARVIS
