# JARVIS Development Roadmap

## Phase Overview

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| 0 | Scaffold | Project skeleton, configs, tests | **Complete** |
| 1 | Inference MVP | Single-model serving with OpenAI API | **Complete** |
| 2 | Router + Multi-Brain | Route queries by domain/difficulty, LoRA swapping | **Complete** |
| 3 | Inference Amplification | Best-of-N, verification, budget forcing | **Complete** |
| 4 | Training (Delta) | ~~Train physics + code brains~~ → HEP LoRA only | **Pivoted** |
| 5 | Specialist Ecosystem | On-demand specialist loading, non-text models | **Complete** |
| 6 | RAG + Knowledge | Physics knowledge base with FAISS retrieval | **Complete** |
| 4M | Model Migration | Migrate from dual 32B bases to Qwen3.5-27B | **New — Not Started** |
| 7 | GRACE Integration | End-to-end HEP agent with JARVIS backend | Not Started |
| 8 | Optimization | Memory tuning, speed profiling, production hardening | Not Started |

---

## Phase 0: Scaffold — COMPLETE

**Goal:** Set up project structure, configs, and test harness.

**Milestone:** `python -m jarvis --help` runs without errors. All configs parse correctly.

---

## Phase 1: Inference MVP — COMPLETE

**Goal:** Serve a single model behind an OpenAI-compatible API.

**Milestone:** `curl -X POST localhost:8000/v1/chat/completions` returns a valid response.

---

## Phase 2: Router + Multi-Brain — COMPLETE

**Goal:** Classify incoming queries and route to the correct brain with LoRA adapter swapping.

**Milestone:** Queries are visibly routed to different domains. `/health` shows active configuration.

---

## Phase 3: Inference Amplification — COMPLETE

**Goal:** Implement the full difficulty-aware inference pipeline.

**Milestone:** Hard physics query → generates 16 candidates → ThinkPRM scores each → budget forcing applied → best solution returned.

---

## Phase 4: Training on Delta — PIVOTED

**Original Goal:** Train custom physics and code brains on NCSA Delta using 8,000 SUs.

**Pivot (2026-04-01):** Qwen3.5-27B (released Feb 2026) exceeds ALL original training targets out-of-the-box:
- GPQA Diamond: 86% (target was 78%)
- LiveCodeBench: 80.7% (target was 65%)
- AIME: 81% (target was 87%, close — inference amplification covers the gap)

**Phase 4A (trace generation) completed** — 5,000 filtered traces from R1-Distill-Qwen-32B. These are for the OLD base model and will not be used for Qwen3.5-27B.

**Phase 4B (SFT) job cancelled** — was training adapter for deprecated R1-Distill-Qwen-32B.

**New Phase 4 scope (HEP LoRA only):**
- [ ] **4A-new: HEP Data Curation (~50 SU)** — Curate HEP physics + HEP code training data from GRACE tool implementations, HEP repos, and existing traces
- [ ] **4B-new: HEP Physics LoRA (~200 SU)** — QDoRA on Qwen3.5-27B with HEP physics data (particle physics, detector design, scintillator properties, kinematics)
- [ ] **4C-new: HEP Code LoRA (~200 SU)** — QDoRA on Qwen3.5-27B with HEP code data (Geant4, ROOT, Pythia8, GDML patterns)
- [ ] **4D-new: General GRPO (optional, ~2,000-3,000 SU)** — RL with verifiable rewards to push general reasoning/coding even higher. Only if baseline evals show room for improvement worth the SU cost.
- [ ] **4E-new: Router Retrain (~50 SU)** — Retrain difficulty classifier using Qwen3.5-27B performance on validation set

**Revised Budget:**
| Phase | SUs | Purpose |
|-------|-----|---------|
| 4A-new | 50 | HEP data curation |
| 4B-new | 200 | HEP physics LoRA |
| 4C-new | 200 | HEP code LoRA |
| 4D-new | 2,000-3,000 | General GRPO (optional) |
| 4E-new | 50 | Router retrain |
| Buffer | ~4,500-5,500 | Future training, new adapters |
| **Spent** | **~76** | Phase 4A traces (old model) |

---

## Phase 4M: Model Migration — NEW

**Goal:** Migrate from dual-base (R1-Distill-Qwen-32B + Qwen2.5-Coder-32B-Instruct) to single Qwen3.5-27B.

**Tasks:**
- [ ] Cancel SFT job 17177608 on Delta
- [ ] Download Qwen3.5-27B to Delta (`/projects/bgde/jhill5/models/qwen3.5-27b`)
- [ ] Verify vLLM compatibility with Qwen3.5 architecture
- [ ] Run baseline evals (GPQA, AIME, LiveCodeBench) to confirm published numbers
- [ ] Update `configs/models.yaml` — single base model entry
- [ ] Update `configs/router.yaml` — remove domain→brain mapping, keep domain classification for specialist/RAG/HEP dispatch
- [ ] Update `configs/deployment.yaml` — single-base memory layout
- [ ] Simplify `src/jarvis/brains/brain_manager.py` — one always-resident base, remove multi-brain resolution
- [ ] Verify ThinkPRM works with Qwen3.5's reasoning format (`<think>` tags)
- [ ] Verify budget forcing conclusion markers match Qwen3.5 output patterns
- [x] Find compatible draft model for speculative decoding → Qwen3.5-0.8B (or native MTP)
- [ ] Update tests to reflect single-base architecture
- [ ] Update all docs (ARCHITECTURE.md, MODELS.md, DEPLOYMENT.md, TRAINING_PIPELINE.md)

**Milestone:** All 142+ tests pass with Qwen3.5-27B as sole base model. Baseline evals confirm published benchmarks.

---

## Phase 5: Specialist Ecosystem — COMPLETE

**Goal:** Enable on-demand loading of domain specialist models.

**Milestone:** Chemistry/protein/DNA queries route to specialist models. Memory never exceeds 128GB.

---

## Phase 6: RAG Knowledge Base — COMPLETE

**Goal:** Build and integrate physics/chemistry/biology knowledge retrieval.

**Milestone:** Physics query about a specific constant or mechanism → retrieved passage prepended → correct answer.

---

## Phase 7: GRACE Integration

**Goal:** End-to-end testing with the GRACE HEP agent using JARVIS as backend.

**Tasks:**
- [ ] Configure GRACE to point at `http://localhost:8000/v1`
- [ ] Test the full GRACE workflow: paper analysis → hypothesis → calculation → code → verification
- [ ] Verify HEP LoRA adapter activation on physics/code queries with HEP content
- [ ] Stress test: run a batch of GRACE tasks and measure throughput, accuracy, and latency
- [ ] Compare against GRACE's previous frontier API results (cost + quality)

**Milestone:** GRACE completes a full HEP analysis task using only JARVIS (no external API calls). Quality within 10% of frontier API baseline.

---

## Phase 8: Optimization

**Goal:** Production hardening, memory optimization, speed tuning.

**Tasks:**
- [ ] Profile memory usage under realistic workloads → identify waste
- [ ] Tune vLLM batch settings for DGX Spark's bandwidth characteristics
- [ ] Implement request queuing for when LoRA swaps are in progress
- [ ] Add logging and monitoring (Prometheus metrics, structured JSON logs)
- [ ] Implement graceful degradation: if DGX Spark thermal throttles → reduce batch size
- [ ] Document operational runbook (startup, monitoring, troubleshooting)
- [ ] (Optional) Set up heterogeneous clustering with spare PC via EXO framework

**Milestone:** JARVIS runs stable for 24+ hours under continuous load. Memory stays within budget. No OOM crashes.
