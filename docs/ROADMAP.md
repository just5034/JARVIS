# JARVIS Development Roadmap

## Phase Overview

| Phase | Name | Goal | Dependencies |
|-------|------|------|-------------|
| 0 | Scaffold | Project skeleton, configs, tests | None |
| 1 | Inference MVP | Single-model serving with OpenAI API | Phase 0 |
| 2 | Router + Multi-Brain | Route queries to correct brain, LoRA swapping | Phase 1 |
| 3 | Inference Amplification | Best-of-N, verification, budget forcing | Phase 2 |
| 4 | Training (Delta) | Train physics + code brains on ACCESS-CI | Phases 0-2 (can overlap with 3) |
| 5 | Specialist Ecosystem | On-demand specialist loading, non-text models | Phase 2 |
| 6 | RAG + Knowledge | Physics knowledge base with FAISS retrieval | Phase 2 |
| 7 | GRACE Integration | End-to-end HEP agent with JARVIS backend | Phases 2-3 |
| 8 | Optimization | Memory tuning, speed profiling, production hardening | All prior phases |

---

## Phase 0: Scaffold

**Goal:** Set up project structure, configs, and test harness.

**Tasks:**
- [ ] Initialize Git repo with project structure per `claude.md`
- [ ] Create `configs/models.yaml` with all model entries (paths, sizes, quantization)
- [ ] Create `configs/inference.yaml` with difficulty-level settings
- [ ] Create `configs/router.yaml` with domain labels and thresholds
- [ ] Write `scripts/download_models.sh` to fetch all required model weights from HuggingFace
- [ ] Set up Python environment (`pyproject.toml` or `requirements.txt`) — key deps: `fastapi`, `uvicorn`, `vllm`, `transformers`, `peft`, `faiss-cpu`, `torch`
- [ ] Write basic test skeleton in `tests/`
- [ ] Write Dockerfile for the JARVIS server

**Milestone:** `python -m jarvis --help` runs without errors. All configs parse correctly.

---

## Phase 1: Inference MVP

**Goal:** Serve a single model (Qwen-32B or any available model) behind an OpenAI-compatible API.

**Tasks:**
- [ ] Implement `src/api/server.py` — FastAPI app with `/v1/chat/completions`, `/v1/models`, `/health`
- [ ] Implement `src/brains/model_loader.py` — Load a single model via vLLM or TensorRT-LLM
- [ ] Support both streaming (SSE) and non-streaming responses
- [ ] Support `temperature`, `max_tokens`, `top_p`, `stop` parameters
- [ ] Implement memory tracking (`/admin/memory` endpoint)
- [ ] Test with a small model first (Qwen-2.5-7B or similar) before scaling to 32B
- [ ] Verify GRACE can connect and issue queries successfully

**Milestone:** `curl -X POST localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}]}'` returns a valid response. GRACE can use JARVIS as a drop-in replacement for OpenAI API.

---

## Phase 2: Router + Multi-Brain

**Goal:** Classify incoming queries and route to the correct brain with LoRA adapter swapping.

**Tasks:**
- [ ] Implement `src/router/domain_classifier.py` — BERT-based domain classifier
- [ ] Implement `src/router/difficulty_estimator.py` — BERT-based difficulty classifier
- [ ] Create training data for router classifiers:
  - Collect/label ~5K examples each for math, physics, code domains
  - Generate difficulty labels by running base model on validation problems
- [ ] Train both classifiers (fine-tune `bert-base-uncased`)
- [ ] Implement `src/brains/adapter_manager.py` — LoRA adapter hot-swapping
  - Load base Qwen-32B once
  - Swap LoRA adapters in milliseconds based on router output
  - Track which adapter is currently active
- [ ] Implement memory-aware loading logic — refuse to load if budget exceeded
- [ ] Add HEP subdomain detection for physics/code queries
- [ ] Integration test: send math, physics, and code queries → verify correct brain handles each

**Milestone:** Queries are visibly routed to different brains. `/health` endpoint shows which adapter is active. Math queries go to math brain, physics to physics brain, code to code brain.

**Note:** For initial development, use off-the-shelf models without custom training. Physics brain = base R1-Distill-Qwen-32B (no adapter), Code brain = same base (no adapter), Math brain = same or R1-Distill-Llama-70B. Custom-trained adapters come in Phase 4.

---

## Phase 3: Inference Amplification

**Goal:** Implement the full difficulty-aware inference pipeline.

**Tasks:**
- [ ] Implement `src/inference/engine.py` — orchestrates strategy selection per difficulty
- [ ] Implement `src/inference/sampling.py` — best-of-N generation with parallel sampling
- [ ] Implement `src/inference/voting.py` — self-consistency majority voting with answer extraction
- [ ] Implement `src/inference/verification.py` — ThinkPRM scoring + pessimistic selection
  - Load ThinkPRM 1.5B as always-resident verifier
  - Score each candidate's reasoning chain
  - Select solution with least uncertainty (pessimistic verification)
- [ ] Implement `src/inference/budget_forcing.py` — "Wait" trick for hard queries
  - Monitor output for premature conclusion tokens
  - Append "Wait" up to 3 times to force continued reasoning
- [ ] Implement `src/inference/verification_chain.py` — append self-check prompts for medium/hard
- [ ] Implement `src/inference/speculative.py` — speculative decoding with 1.5B draft model
- [ ] Implement `src/inference/code_verifier.py` — S* execution-based verification for code brain
  - Sandboxed Python executor (Docker container)
  - Generate distinguishing test inputs
  - Execute candidates, select by correct behavior
- [ ] Wire difficulty level from router → inference engine strategy selection
- [ ] Configurable via `configs/inference.yaml`

**Milestone:** Hard physics query → generates 16 candidates → ThinkPRM scores each → budget forcing applied → best solution returned. Hard code query → S* verification with execution. Easy queries still fast (single pass).

---

## Phase 4: Training on Delta (ACCESS-CI)

**Goal:** Train custom physics and code brains on NCSA Delta using 8,000 ACCESS-CI SUs.

**This phase runs in parallel with Phases 3, 5, 6 on the deployment side.**

**Subphases:** (See `docs/TRAINING_PIPELINE.md` for full details)

- [ ] **4A: Physics Data Generation (350 SU)**
  - Multi-teacher trace generation from R1-0528
  - LADDER curriculum generation
  - Rejection sampling + quality filtering
  - Synthetic textbook chapters
- [ ] **4B: Physics Distillation SFT (800 SU)**
  - QDoRA (r=32) on R1-Distill-Qwen-32B with 100K traces
  - Eval on GPQA Diamond → target 68-72%
- [ ] **4C: Physics Curriculum GRPO (2,000 SU)**
  - Staged difficulty RL with multi-signal rewards + PRIME
  - Eval on GPQA Diamond → target 72-78%
- [ ] **4D: Physics ETTRL Polish (900 SU on H200)**
  - Test-time RL on 500 hard unlabeled problems
  - Eval on GPQA Diamond → target 74-80%
- [ ] **4E: Physics Post-Processing (450 SU)**
  - POME, checkpoint merging, one self-distillation cycle
- [ ] **4F: Code AZR Self-Play (2,000 SU)**
  - Early 32B validation (200 SU from buffer)
  - Full AZR training if validation passes; GRPO fallback if not
  - Eval on LiveCodeBench → target 55-65%
- [ ] **4G: Code Targeted SFT + Post-Processing (600 SU)**
  - QDoRA on competition problems, POME, merge, self-distillation
- [ ] **4H: Router Training (200 SU)**
  - Generate difficulty labels from trained brains
  - Train domain + difficulty classifiers
- [ ] **4I: Export trained adapters** → Copy to DGX Spark deployment

**Milestone:** Physics brain GPQA ≥ 78%. Code brain LiveCodeBench ≥ 65%. Math brain AIME ≥ 87% (off-shelf + inference amplification). Adapters exported for deployment.

---

## Phase 5: Specialist Ecosystem

**Goal:** Enable on-demand loading of domain specialist models.

**Tasks:**
- [ ] Implement `src/specialists/registry.py` — parse `configs/models.yaml` specialist entries
- [ ] Implement `src/specialists/loader.py` — on-demand model loading from SSD with LRU eviction
- [ ] Implement specialist API adapters for non-text models:
  - [ ] `src/specialists/adapters/esm3.py` — protein sequence/structure input → ESM3 → structured output
  - [ ] `src/specialists/adapters/evo2.py` — DNA sequence input → Evo 2 → variant/generation output
  - [ ] `src/specialists/adapters/text_llm.py` — standard chat format (ChemLLM, BioMistral, SaulLM, etc.)
- [ ] Extend router domain classifier to include specialist domains
- [ ] Add memory-aware eviction: when loading a new specialist would exceed budget, evict LRU specialist
- [ ] Integration test: chemistry query → ChemLLM loaded from SSD → response → ChemLLM evicted when physics query arrives

**Milestone:** Send a chemistry question → ChemLLM-7B loads in <10s → correct response. Send a protein sequence → ESM3 returns structure prediction. Memory never exceeds 128GB.

---

## Phase 6: RAG Knowledge Base

**Goal:** Build and integrate physics/chemistry/biology knowledge retrieval.

**Tasks:**
- [ ] Curate reference corpus:
  - Physical constants and equations
  - Key derivations and proofs
  - Reaction mechanisms and molecular properties
  - Unit conversion tables
  - HEP-specific: cross-sections, PDG data, detector parameters
- [ ] Embed corpus with `all-MiniLM-L6-v2` into FAISS index
- [ ] Implement `src/rag/retriever.py` — query embedding + top-K retrieval
- [ ] Implement `src/rag/augmenter.py` — prepend retrieved passages to prompt
- [ ] Only activate for physics-domain queries (router signals)
- [ ] Evaluate RAG impact: run GPQA with and without RAG → measure delta

**Milestone:** Physics query about a specific constant or mechanism → retrieved passage prepended → correct answer that the model would otherwise miss.

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
- [ ] Tune vLLM/TensorRT-LLM batch settings for DGX Spark's bandwidth characteristics
- [ ] Implement request queuing for when model swaps are in progress
- [ ] Add logging and monitoring (Prometheus metrics, structured JSON logs)
- [ ] Implement graceful degradation: if DGX Spark thermal throttles → reduce batch size
- [ ] Document operational runbook (startup, monitoring, troubleshooting)
- [ ] (Optional) Set up heterogeneous clustering with spare PC via EXO framework

**Milestone:** JARVIS runs stable for 24+ hours under continuous load. Memory stays within budget. No OOM crashes.
