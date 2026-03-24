# JARVIS Master Plan v5: Unified Architecture, Specialist Ecosystem & Local Deployment

**A routed multi-specialist system — not an MoE, but a plug-and-play ensemble of independently trained brains behind a shared API, designed for extensibility and local deployment.**

**Training Budget: 8,000 ACCESS-CI SUs | Training Hardware: Delta A100 40GB (1× rate) + H200 141GB (3× rate)**
**Deployment Target: NVIDIA DGX Spark (128GB unified, GB10 Grace Blackwell) | ~$4,700**

---

## 1. System Architecture

```
                         ┌──────────────────────┐
          User Query ──▶ │  DIFFICULTY-AWARE     │
          (via GRACE)    │  ROUTER               │
                         │  (domain + hardness)   │
                         └───┬──────┬──────┬──┬──┘
                             │      │      │  │
             Easy/Med/Hard   │      │      │  │  Additional domains
                  ┌──────────┘      │      │  └──────────────┐
                  ▼                 ▼      ▼                  ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
         │  MATH BRAIN  │  │ PHYSICS BRAIN│  │  CODE BRAIN  │  │  SPECIALIST RACK │
         │              │  │              │  │              │  │  (hot-swappable)  │
         │ R1-Distill   │  │ Custom 32B   │  │ Custom 32B   │  │                  │
         │ 70B          │  │ Distill+RL   │  │ AZR Self-Play│  │ ESM3 (proteins)  │
         │ Off-shelf    │  │              │  │              │  │ Evo 2 (genomics) │
         │ 0 SU train   │  │ + RAG        │  │              │  │ ChemLLM (chem)   │
         │              │  │ ~4,200 SU    │  │ ~2,800 SU    │  │ BioMistral (med) │
         │              │  │              │  │              │  │ + future domains │
         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────────┘
                │                 │                  │                 │
                └────────┬────────┘──────────────────┘─────────────────┘
                         ▼
              ┌─────────────────────┐
              │  INFERENCE ENGINE   │
              │                     │
              │  Easy: 1 pass       │
              │  Med:  best-of-4    │
              │  Hard: best-of-16   │
              │        + PRM/Heimdall│
              │        + verify chain│
              │        + ETTRL (opt) │
              │                     │
              │  Speculative decode  │
              │  for speed           │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  DEPLOYMENT TARGET  │
              │                     │
              │  NVIDIA DGX Spark   │
              │  128GB unified RAM  │
              │  4TB NVMe storage   │
              │  GB10 Blackwell GPU │
              │  OpenAI-compatible  │
              │  API @ localhost    │
              └─────────────────────┘
```

**Key changes from v2:**
- Router now estimates difficulty, not just domain → adaptive inference compute
- Code brain uses Absolute Zero self-play RL instead of standard GRPO → unlimited self-generated training data
- Physics brain adds RAG knowledge base → partially closes pre-training knowledge gap at zero training cost
- ETTRL added as optional final polish phase → model continues improving on hard unlabeled problems
- Heimdall-style pessimistic verification replaces generic PRM plan → validated 54.2%→83.3% on AIME with R1-Distill-Qwen-32B
- Speculative decoding enables heavier test-time scaling without latency penalty

**Key changes from v3 (audit-verified):**
- Physics GPQA projection revised down from 82-87% to **78-84%** — Heimdall validated on math, not science; transfer to GPQA uncertain
- Heimdall now distinguishes **practical deployment** (16× compute → 70%) from **batch ceiling** (N=64, M=256 → 83.3%)
- AZR flagged as unverified at 32B scale (tested only up to 14B); fallback to standard GRPO if scaling issues emerge
- AZR uses REINFORCE++ (not GRPO) — same veRL framework but different algorithm
- ETTRL budget increased from 600 to 900 SU to account for tight H200 memory at 32B scale
- **Budget Forcing** ("Wait" trick) added — zero-cost inference technique, s1 paper exceeded o1-preview on AIME by 27%
- **S\* test-time scaling** added for code brain — pushes R1-Distill-Qwen-32B to 85.7% on LiveCodeBench
- NTele-R1-32B validates that just 0.8K curated examples can reach 81.87% AIME — confirms quality-over-quantity data strategy

**Key changes in v5:**
- **Architecture clarity:** JARVIS is a routed ensemble / plug-and-play specialist system — NOT a Mixture of Experts. Independent models, external router, no shared weights between brains
- **Specialist rack added:** Extensible domain plug-in system for deploying additional specialist models (proteins, genomics, chemistry, biomedicine, legal, finance, etc.) via LoRA adapters or standalone lightweight models
- **Deployment target specified:** NVIDIA DGX Spark ($4,699, 128GB unified RAM, GB10 Grace Blackwell) as primary local deployment hardware. R1-0528 (685B) confirmed as NOT deployable locally — math brain defaults to R1-Distill-Llama-70B or R1-Distill-Qwen-32B
- **Multi-device clustering documented:** DGX Spark supports official 2-unit clustering (256GB) and up to 4-unit (512GB). Heterogeneous clustering with consumer GPUs possible via EXO Labs framework

---

## 2. Math Brain (0 SU Training Investment)

**Model:** DeepSeek-R1-0528 (685B MoE) if inference infra supports it; otherwise R1-Distill-Llama-70B.

| Benchmark | R1-0528 | R1-Distill-70B | Frontier |
|-----------|---------|----------------|----------|
| MATH-500 | 97.3% | 94.5% | ~98% |
| AIME 2024 | 79.8% | 70.0% | 79.8% (o1) |
| AIME 2025 | 87.5% | — | 100% (GPT-5.2) |
| GPQA Diamond | 81.0% | 65.2% | 94.3% |

**Inference amplification (free):** Self-consistency voting (consensus@16) + Heimdall verification on hard problems. R1-Zero demonstrated 71%→86.7% on AIME from consensus alone.

**No training needed.** Math is solved.

---

## 3. Physics Brain (~4,200 SU)

### 3.1 Starting Point & Target

| Stage | GPQA Diamond | Notes |
|-------|-------------|-------|
| Base: R1-Distill-Qwen-32B | 62.1% | Fits A100 40GB, 1× SU rate |
| Human PhD experts | 65-74% | Benchmark reference |
| After distillation + RL | ~72-78% | Training-time target |
| After POME + merge | ~73-79% | Free post-processing |
| After best-of-16 + verification | ~76-82% | Inference-time scaling (⚠️ see note) |
| After ETTRL polish | ~78-84% | Final ceiling |
| R1-0528 (685B MoE) | 81.0% | Comparison: we approach this with a 32B |
| Kimi K2 Thinking | ~91.3% | Stretch comparison |

**⚠️ IMPORTANT CAVEAT ON VERIFICATION GAINS:** Heimdall's 54.2%→83.3% result was validated on **AIME (math)**, not GPQA (science). ThinkPRM showed +8% on a GPQA subset and +4.5% on LiveCodeBench, suggesting verification transfers across domains but with reduced magnitude. Our physics verification gain estimate is **+3-6 points** (conservative) rather than the +16-29 points Heimdall achieves on math. Science verification is harder because incorrect reasoning can lead to plausible-sounding wrong answers that are difficult to distinguish from correct ones.

**Practical vs Batch deployment:** Heimdall's 83.3% AIME result requires N=64 solutions × M=256 verifications = 16,384 model calls per problem. At deployment, we target the **16× compute tier** (N=16 solutions with verification → 70% on AIME equivalent), which is practical for interactive use (~30-60 seconds per hard query on dual 5090s).

### 3.2 Training Pipeline (Chronological)

**Phase A: Data Generation (~350 SU)**

1. **Multi-teacher trace generation (~200 SU):** Run R1-0528 (via API or quantized local inference) on 50K physics/chemistry/biology problems. Generate 8 traces per problem. Sources: GPQA train split, SciInstruct, graduate textbook problems, Olympiad physics, ARC-Challenge, arXiv problem sets.

2. **LADDER-style curriculum generation (~50 SU):** For the hardest 5K problems, have the model recursively generate easier variants. This creates a natural difficulty ladder: variant_easy → variant_medium → original_hard. LADDER pushed a 7B model to 90% on MIT Integration Bee when combined with TTRL.

3. **Rejection sampling + quality filtering (~50 SU):** From 400K raw traces, filter to ~100K high-quality examples using: correctness verification (answer matches ground truth), LLM judge for reasoning quality (AskLLM), diversity filtering (remove near-duplicates), conciseness preference.

4. **Synthetic textbook chapters (~50 SU):** Generate ~5K "textbook-style" explanations from R1-0528 covering core physics, chemistry, and biology concepts. These teach conceptual understanding, not just problem-solving patterns.

**Phase B: Distillation SFT (~800 SU)**

Fine-tune R1-Distill-Qwen-32B on the curated 100K-trace dataset using QDoRA (quantized DoRA, r=32). Training on 4× A100 40GB with DeepSpeed ZeRO-3.

- Mix: 70% problem-solving traces, 20% textbook explanations, 10% general instruction data (prevents catastrophic forgetting)
- Expected gain: 62% → ~68-72% on GPQA Diamond

**Phase C: Curriculum GRPO RL (~2,000 SU)**

Apply GRPO with staged difficulty and multi-signal rewards:

- **Curriculum:** Stage 1 (steps 1-100) undergrad-level → Stage 2 (100-300) graduate-level → Stage 3 (300-500+) competition/research-level
- **Rewards:** Numerical answer verification + unit/dimensional check + reasoning structure reward + length penalty
- **PRIME implicit rewards:** Extract process-level reward signals from reference model log-probabilities for better credit assignment on long derivations. Zero additional compute.
- **Self-refinement:** Two-pass generation (solve → critique → correct). Reward applied to corrected answer. ~2× generation cost per problem, but trains internal error-correction.

Expected gain: ~68-72% → ~72-78% on GPQA Diamond

**Phase D: ETTRL Final Polish (~900 SU on H200 = 300 effective H200 GPU-hrs)**

⚠️ This phase runs on H200 nodes (3× rate) because GRPO weight updates during inference require the full model + optimizer states in memory — exceeding the 40GB available on Delta A100s for a 32B model.

Apply ETTRL (Entropy-Based Test-Time RL) on a curated set of ~500 hard physics problems without ground-truth labels. The model generates 64 solutions per problem, uses majority voting as pseudo-labels, and runs GRPO updates. ETTRL branches only at high-entropy tokens, halving compute vs standard TTRL.

- TTRL boosted Qwen-2.5-Math-7B by 211% on AIME (v3 paper). For our already-strong 32B model, gains will be smaller but still meaningful.
- Runs for 40-60 episodes.
- Budget increased from original 600 SU estimate to 900 SU to account for the tight memory situation on H200 and potential need for gradient checkpointing overhead.
- Expected gain: +2-4% on hard physics problems

**Phase E: Post-Training (0 SU)**

1. **POME:** SVD projection on weight deltas. +1-2.5% free.
2. **Checkpoint merging:** SLERP-merge best 3-5 RL checkpoints. +0.5-1% generalization.
3. **One cycle self-distillation (~450 SU):** Take model's best outputs on training set, filter for quality, run one additional SFT pass. +1-3%.

### 3.3 RAG Knowledge Base (0 SU — Engineering Only)

Build a vector database of physics/chemistry/biology reference material:
- Physical constants and equations
- Key derivations and proofs
- Reaction mechanisms and molecular properties
- Unit conversion tables and dimensional analysis patterns

At inference time, the router retrieves relevant passages and prepends them to the query. This compensates for factual knowledge gaps that fine-tuning cannot fill. Expected: +2-5% on knowledge-heavy questions.

### 3.4 Physics Brain Budget Summary

| Phase | SUs | Effective GPU-hrs | System |
|-------|-----|-------------------|--------|
| Data generation | 350 | 350 A100-hrs | A100 |
| Distillation SFT | 800 | 800 A100-hrs | A100 |
| Curriculum GRPO RL | 2,000 | 2,000 A100-hrs | A100 |
| ETTRL polish | 600 | 200 H200-hrs | H200 (3×) |
| Self-distillation cycle | 450 | 450 A100-hrs | A100 |
| **Total** | **4,200** | | |

---

## 4. Code Brain (~2,800 SU)

### 4.1 Starting Point & Target

| Stage | HumanEval | LiveCodeBench | Notes |
|-------|-----------|---------------|-------|
| Base: Qwen3-32B | ~85% | ~45% | Strong general base |
| After AZR self-play RL | ~90-93% | ~55-65% | Training-time target |
| After POME + merge | ~91-94% | ~56-66% | Free post-processing |
| After Heimdall best-of-16 | ~93-95% | ~60-70% | Inference-time |
| Frontier | ~97% | ~76% | Comparison |

### 4.2 Training Pipeline

**Phase A: Absolute Zero Self-Play RL (~2,200 SU)**

This is the major innovation for the code brain. Instead of standard GRPO on curated datasets, use the **Absolute Zero Reasoner** paradigm (NeurIPS 2025 Spotlight):

- The model simultaneously **proposes** coding challenges and **solves** them
- A Python executor validates both the proposed tasks and the solutions
- The proposer is rewarded for generating problems at the model's learning frontier (hard enough to be informative, easy enough to occasionally solve)
- The solver is rewarded for correct solutions verified by code execution
- Three reasoning types: abduction (given output, find input), deduction (given input, find output), induction (given examples, find the rule)
- **Algorithm:** AZR uses REINFORCE++ (not GRPO) with a custom multi-task advantage estimator. Both run on the veRL framework, so infrastructure is shared.

**Why AZR over standard GRPO for code:**
- Unlimited self-generated training data — no dataset bottleneck
- Difficulty automatically calibrates to the model's current level (built-in curriculum)
- Achieves SOTA among zero-setting models, beating those trained on tens of thousands of human-curated examples
- Open-source implementation on veRL framework, runs on A100s
- The code executor provides perfectly verifiable rewards — the ideal domain for AZR
- Scaling trend: 3B → 7B → 14B showed +5.7, +10.2, +13.2 point gains — bigger models benefit more

**⚠️ RISK: AZR has only been validated up to 14B parameters.** The 32B scale is extrapolation based on the clear scaling trend. The veRL framework supports 32B models, and INTELLECT-2 has validated GRPO at 32B scale, but AZR's specific propose-and-solve dynamics have not been tested at this scale. **Mitigation:** Budget 200 SUs of the exploration buffer for early AZR validation at 32B. If scaling issues emerge (unstable training, mode collapse in proposer), fall back to standard GRPO with curated code datasets — the infrastructure is identical.

**Implementation:** Use the AZR codebase (fork of veRL), seed with a single identity-function triplet, train for ~2,000 A100 GPU hours on 4× A100 nodes. Reserve 200 SU for fallback contingency.

**Phase B: Targeted SFT on Hard Patterns (~300 SU)**

After AZR, the model is strong at algorithmic reasoning but may miss domain-specific patterns (data structures, systems programming, specific libraries). Run a focused SFT pass on:
- LiveCodeBench problems with R1-0528 solutions (rejection sampled)
- Competition programming problems (CodeContests, APPS) with verified solutions
- ~30K high-quality examples, QDoRA fine-tuning

**Phase C: Post-Training (0 SU)**

1. POME projection on weight deltas
2. Checkpoint merging of best AZR checkpoints
3. (Optional) One self-distillation cycle (~300 SU if budget allows)

### 4.3 Code Brain Budget Summary

| Phase | SUs | Effective GPU-hrs | System |
|-------|-----|-------------------|--------|
| AZR self-play RL | 2,200 | 2,200 A100-hrs | A100 |
| Targeted SFT | 300 | 300 A100-hrs | A100 |
| Self-distillation (opt) | 300 | 300 A100-hrs | A100 |
| **Total** | **2,800** | | |

---

## 5. Difficulty-Aware Router (~200 SU)

### 5.1 Two-Dimensional Routing

The router makes two decisions per query:

1. **Domain classification:** Math, Physics, or Code (BERT classifier, ~5K labeled examples per class)
2. **Difficulty estimation:** Easy, Medium, or Hard (separate BERT classifier trained on model failure data)

Difficulty labels are generated automatically: run each specialist on a validation set, problems answered correctly in one pass = Easy, correct with best-of-4 = Medium, incorrect or only correct with best-of-16 = Hard.

### 5.2 Adaptive Inference Compute

| Difficulty | Strategy | Inference Cost | When |
|------------|----------|----------------|------|
| Easy | Single pass | 1× | ~60% of queries |
| Medium | Best-of-4 + self-consistency | 4× | ~25% of queries |
| Hard | Best-of-16 + Heimdall PRM + verification chain + extended thinking (32K+) | 16-32× | ~15% of queries |

### 5.3 Cross-Domain Routing

For multi-domain queries (e.g., "derive and implement this physics simulation"):
- Router detects both physics and code signals
- Sequential pipeline: Physics brain generates mathematical framework → Code brain implements it
- Confidence-based: if either specialist outputs low confidence, escalate to heavy verification

---

## 6. Inference Engine

### 6.1 Heimdall-Style Pessimistic Verification

The strongest validated result from our research: Heimdall (trained with pure RL for math verification) achieved 94.5% verification accuracy on competitive math. Combined with Pessimistic Verification on R1-Distill-Qwen-32B:

| Compute Budget | AIME 2025 Score | Practical? |
|---------------|-----------------|------------|
| 1× (single pass) | 54.2% | Yes — real-time |
| 16× (N=16 solutions + verification) | 70.0% | Yes — 30-60s on dual 5090s |
| Max (N=64, M=256) | 83.3% | Batch only — 30-60min per problem |

**⚠️ Domain transfer caveat:** These numbers are for **math (AIME)**. Verification on **physics (GPQA)** and **code (LiveCodeBench)** will show smaller gains because: (a) incorrect science reasoning can be internally consistent and hard to verify, (b) ThinkPRM showed +8% on GPQA and +4.5% on LiveCodeBench vs Heimdall's +16-29 points on AIME. Our inference projections use the conservative estimates.

**Implementation for JARVIS:**
- Use ThinkPRM (1.5B generative PRM) as the base verifier — zero training cost, works zero-shot
- Apply Pessimistic Verification: frame as multi-armed bandit, select solution with least uncertainty
- **Math brain:** Full Heimdall pipeline (16× practical, batch ceiling available)
- **Physics brain:** ThinkPRM best-of-16 with pessimistic selection (+3-6 pts expected)
- **Code brain:** S\*-style verification with execution-based distinguishing inputs (+4-8 pts expected)

### 6.2 Verification Chains

After initial generation on medium/hard problems, append: "Verify your answer by [substituting back / checking dimensional analysis / testing edge cases]. If you find an error, correct it."

Catches ~10-20% of residual errors. Cost: ~1.5× per query.

### 6.3 Speculative Decoding

Use R1-Distill-Qwen-1.5B as a draft model to accelerate inference 2-3×. This enables heavier test-time scaling (more best-of-N samples) within acceptable latency.

### 6.4 Extended Thinking Budgets

- Easy: 4K token thinking budget
- Medium: 16K token thinking budget
- Hard: 32-64K token thinking budget
- Physics derivations: up to 128K (requires FP8 KV cache + SSD offload)

### 6.5 Context Window & KV Cache Management — NEW in v5

**The problem:** Every token in context generates KV cache entries that consume RAM alongside model weights. For a 32B model at FP16, each token costs ~0.25 MB of KV cache. At 128K context with best-of-16 parallel, that's 16 × 32 GB = 512 GB — far exceeding the DGX Spark's 128 GB.

**Solution: stack KV cache optimizations to extend effective context.**

| Technique | Memory Reduction | Quality Loss | Implementation |
|-----------|-----------------|-------------|----------------|
| FP8 KV cache | 2× | ~Negligible | vLLM flag: `kv_cache_dtype="fp8"` |
| 2-bit KV (KVQuant/AQUA-KV) | 6-8× | <1% perplexity | llm-compressor calibration (1-6 hrs one-time) |
| SSD offload (KVSwap) | Effectively unlimited (single stream) | Small latency (~ms/layer) | Config change, uses 4TB NVMe |
| StreamingLLM eviction | Fixed-size rolling window | Loses mid-context detail | For multi-turn conversations only |
| RAG (already planned) | N/A — replaces context | Retrieval quality dependent | FAISS, §3.3 |
| Context compression | 3-10× on conversation history | Lossy on detail | Software summarization pass |

**Practical context limits on DGX Spark (Config A: ~66 GB available after core system + specialists):**

| Difficulty | Sampling | KV Config | Max Context |
|-----------|----------|-----------|-------------|
| Easy | 1 pass | FP8 | 128K |
| Medium | best-of-4 parallel | FP8 | 64K |
| Hard | best-of-16 parallel | FP8 | 32K |
| Hard | best-of-16 parallel | 2-bit KVQuant | 64K |
| Hard | best-of-16 sequential | FP8 | 128K |
| Derivation | 1 pass + SSD offload | FP8 + offload | 128K+ |

**Default configuration:** FP8 KV cache enabled globally. 2-bit KVQuant enabled for hard queries. SSD offload available for physics derivation mode.

### 6.6 Budget Forcing ("Wait" Trick) — NEW in v4

**What:** When the model tries to end its reasoning prematurely on a hard problem, append "Wait" to force continued thinking. The model re-examines its answer, often catching errors.

**Why it works:** The s1 paper demonstrated that SFT on just 1,000 curated examples + budget forcing exceeded o1-preview on AIME by 27%. The "Wait" trick is essentially free — it's a prompt-level intervention that forces the model to double-check. When a model tries to conclude with a wrong answer, the forced continuation often triggers self-correction.

**Implementation:** For hard queries (as flagged by the difficulty router), monitor the model's output. If it produces a conclusion token before reaching the thinking budget, append "Wait" up to 3 times to encourage re-examination. This is implemented at the inference server level, zero training cost.

**Cost:** 0 (inference-time prompt intervention). Adds ~1.3× tokens per hard query.

### 6.7 S* Test-Time Scaling for Code — NEW in v4

**What:** Sequential + parallel scaling with adaptive distinguishing input generation. The system generates multiple candidate solutions, then automatically creates test inputs that distinguish between them, using execution results to identify the correct solution.

**Why it works:** S* pushed R1-Distill-Qwen-32B to **85.7% on LiveCodeBench**, approaching o1 (88.5%). It outperforms both majority voting and standard best-of-N by actively probing for behavioral differences between candidates rather than passively scoring them. For code, this is especially powerful because you can actually *run* the distinguishing inputs.

**Implementation:** For code brain hard queries: generate 16 candidate solutions → generate distinguishing test inputs → execute all candidates against distinguishing inputs → select the solution with correct behavior. The code executor provides ground-truth distinguishing signals that no verifier model can match.

**Cost:** 0 (inference-time only). Requires code execution environment at deployment.

---

## 7. Complete Budget Allocation

| Phase | SUs | % of Budget | Activity |
|-------|-----|-------------|----------|
| **Physics: Data gen** | 350 | 4.4% | Multi-teacher traces, LADDER curriculum, filtering, textbooks |
| **Physics: Distillation SFT** | 800 | 10.0% | QDoRA on R1-Distill-Qwen-32B with 100K traces |
| **Physics: Curriculum GRPO** | 2,000 | 25.0% | Staged difficulty RL with multi-signal rewards + PRIME |
| **Physics: ETTRL polish** | 900 | 11.3% | Test-time RL on 500 hard unlabeled problems (H200, 3×) |
| **Physics: Self-distillation** | 450 | 5.6% | One cycle: best outputs → SFT |
| **Code: AZR self-play RL** | 2,000 | 25.0% | Absolute Zero proposer-solver loop (REINFORCE++) |
| **Code: Targeted SFT** | 300 | 3.8% | QDoRA on LiveCodeBench + competition problems |
| **Code: Self-distillation** | 300 | 3.8% | One cycle: best outputs → SFT |
| **Router + evaluation** | 200 | 2.5% | Train BERT classifiers, run benchmark suite |
| **Exploration + buffer** | 700 | 8.8% | Hyperparameter sweeps, AZR 32B validation, failed runs |
| **TOTAL** | **8,000** | **100%** | |

**Budget changes from v3:** ETTRL increased 600→900 SU (memory safety margin on H200). Code AZR decreased 2,200→2,000 SU (200 SU moved to exploration buffer for AZR 32B validation). Buffer decreased 800→700 SU (absorbed ETTRL increase). Net: still 8,000 SU total.

---

## 8. Techniques Evaluated and Excluded

| Technique | Why Excluded |
|-----------|-------------|
| Continual pre-training | 800-1,500 SU for uncertain gains; distillation from R1-0528 transfers knowledge more efficiently |
| Full adversarial self-play critic | Complexity not justified; self-refinement training + PRM verification captures similar benefit |
| Mixture of LoRA experts | Save for Phase 2 if single physics adapter plateaus; adds complexity before baseline is established |
| Multiple self-distillation cycles | Diminishing returns after first cycle; not worth 2× budget for <1% marginal gain |
| Muon optimizer for fine-tuning | Optimizer mismatch with AdamW-pretrained checkpoints degrades performance; only useful for training from scratch |
| 70B model for physics | A100 40GB constraint forces H200 at 3× rate; 32B gives 3× more RL iterations for same SU budget |

---

## 9. Every Free/Near-Free Technique (Applied Unconditionally)

These cost zero or near-zero SUs and are applied to every model we produce:

| Technique | Cost | Effect | Applied To |
|-----------|------|--------|------------|
| POME post-processing | 0 (CPU) | +1-2.5% across benchmarks | All fine-tuned models |
| Checkpoint SLERP merging | 0 (CPU) | +0.5-1% generalization | All models after RL |
| DoRA over LoRA | 0 (same compute) | +1-3.7% consistent | All SFT runs |
| PRIME implicit rewards | 0 (uses ref model) | Better credit assignment | All GRPO runs |
| Self-consistency voting | 0 (inference) | +5-15% on hard problems | All specialists at deployment |
| ThinkPRM verification | 0 (off-shelf 1.5B) | +3-8% on hard problems (domain-dependent) | All specialists at deployment |
| Verification chains | 0 (prompt) | Catches 10-20% errors | Medium/hard queries |
| Extended thinking | 0 (config) | Scales with compute | Hard queries |
| Budget forcing ("Wait") | 0 (prompt) | Forces self-correction on hard problems | Hard queries (all brains) |
| S* execution-based verification | 0 (inference) | +4-8% on code with distinguishing inputs | Code brain hard queries |
| Speculative decoding | 0 (small draft model) | 2-3× faster inference | All inference |
| RAG for physics | 0 (engineering) | +2-5% on knowledge questions | Physics brain |
| Curriculum scheduling | 0 (changes order) | +2-3% stability | All RL runs |
| Multi-signal rewards | 0 (reward design) | More robust RL | All RL runs |
| FP8 KV cache quantization | 0 (vLLM flag) | 2× context capacity | All inference (default on) |
| 2-bit KV quantization (KVQuant) | 0 (one-time calibration) | 6-8× context capacity | Hard queries with long context |
| KV cache SSD offload | 0 (config) | Effectively unlimited single-stream context | Physics derivation mode |
| Context compression | 0 (software) | 3-10× on conversation history | Long GRACE workflows |

---

## 10. Projected Performance Summary

### Physics (GPQA Diamond)

| Method | Score | Cumulative Gain |
|--------|-------|-----------------|
| Off-the-shelf R1-Distill-Qwen-32B | 62.1% | Baseline |
| + Multi-teacher distillation + QDoRA SFT | ~68-72% | +6-10 pts |
| + Curriculum GRPO + PRIME + self-refinement | ~72-78% | +4-6 pts |
| + ETTRL on hard problems | ~74-80% | +2-3 pts |
| + POME + checkpoint merge + self-distillation | ~75-81% | +1-2 pts |
| + Best-of-16 + ThinkPRM verification | ~78-84% | +3-4 pts (⚠️ conservative — see §6.1) |
| + RAG + verification chains + budget forcing | ~79-84% | +0-1 pts |
| **JARVIS Physics Brain** | **~78-84%** | At or near R1-0528 (81%) |

**Revised from v3:** Previous projection was 82-87%. Reduced to 78-84% because Heimdall verification gains are validated on math, not science. ThinkPRM shows smaller (+8%) gains on GPQA subset vs Heimdall's +16-29 points on AIME. The range is honest — the lower end (78%) is achievable with high confidence, the upper end (84%) requires everything to go well.

### Code (LiveCodeBench)

| Method | Score | Cumulative Gain |
|--------|-------|-----------------|
| Off-the-shelf Qwen3-32B | ~45% | Baseline |
| + AZR self-play RL (⚠️ unverified at 32B, see §4.2) | ~55-62% | +10-17 pts |
| + Targeted SFT on competition problems | ~58-65% | +3-4 pts |
| + POME + checkpoint merge + self-distillation | ~59-66% | +1-2 pts |
| + S* execution-based verification + budget forcing | ~65-72% | +5-7 pts |
| **JARVIS Code Brain** | **~65-72%** | Competitive with frontier |

**Revised from v3:** Range widened slightly upward (was 63-70%) due to S* addition. S* specifically pushed R1-Distill-Qwen-32B to 85.7% on LiveCodeBench — though that's a stronger base model than our code-specialized Qwen3-32B, the technique applies directly.

### Math (AIME 2024)

| Method | Score | Notes |
|--------|-------|-------|
| R1-0528 (off-shelf) | 79.8% | Matches o1 |
| + Consensus@16 | ~87% | Free inference scaling |
| + Heimdall verification | ~90%+ | Upper bound |
| **JARVIS Math Brain** | **~87-90%** | Frontier-competitive |

---

## 11. Extensible Specialist Ecosystem — NEW in v5

### 11.1 Architecture: Why JARVIS Is Not an MoE

JARVIS is a **routed ensemble** (or "mixture of specialists system"), not a true Mixture of Experts:

- **True MoE** (e.g., DeepSeek-V3, Mixtral): A single model with shared attention layers, multiple expert FFN blocks, and a learned **token-level** router — all trained end-to-end in one set of weights. One artifact.
- **JARVIS**: Completely separate, independently trained models orchestrated by an external classifier operating at the **query level**. No shared layers, no shared weights, no shared training signal.

This distinction is the key architectural advantage: any brain can be swapped, upgraded, retrained, or extended without touching the others. The system is a **plug-and-play specialist rack** by design.

### 11.2 Adding New Specialist Domains

Adding a new specialist (e.g., chemistry, biology, legal) requires three steps:

1. **Router expansion:** Add a new domain label to the classifier output space. Retrain/fine-tune on examples of the new domain. The router is a lightweight BERT classifier — retraining is cheap (<50 SU).
2. **New brain deployment:** Either (a) train a new LoRA adapter on the existing Qwen-32B base (a few hundred MB, hot-swappable in milliseconds), or (b) deploy a standalone specialist model loaded from SSD on demand (5-10 seconds for 7B models, 30-60 seconds for 70B).
3. **Inference layer:** The difficulty-aware amplification (best-of-N, verification, budget forcing) is domain-agnostic and applies automatically to any new specialist.

GRACE integration requires zero changes — GRACE calls JARVIS through an OpenAI-compatible API at localhost. New specialists are invisible to the client.

### 11.3 Candidate Specialist Models (Researched)

These open-source specialized models have been identified as deployable within JARVIS. Models are grouped by category:

**Biological Foundation Models (non-text, domain-native)**

| Model | Domain | Params | FP4 Size | Key Capability | License |
|-------|--------|--------|----------|---------------|---------|
| ESM3-open | Proteins | 1.4B | ~0.7 GB | Joint sequence/structure/function reasoning. Generated novel GFP equivalent to 500M years of evolution | Open (HuggingFace) |
| Evo 2 | Genomics/DNA | 7B / 40B | 3.5 / 20 GB | Single-nucleotide DNA modeling across all life. Generated a functional bacteriophage. Published Nature Mar 2026 | Apache 2.0 |
| ESM-2 / ESMFold | Protein structure | Up to 15B | ~7.5 GB | Structure prediction 10× faster than AlphaFold2. No MSA required | MIT / Apache 2.0 |

**Domain-Specialized Text LLMs**

| Model | Domain | Params | FP4 Size | Key Result | License |
|-------|--------|--------|----------|------------|---------|
| ChemLLM-7B | Chemistry | 7B | ~3.5 GB | First chemistry-dedicated LLM. GPT-4 competitive on ChemBench (9 tasks) | Apache 2.0 |
| BioMistral-7B | Biomedicine | 7B | ~3.5 GB | Outperforms open medical models on 10 QA tasks. Multilingual (8 languages) | Open |
| OpenBioLLM-70B | Biomedicine | 70B | ~35 GB | 86.06% avg across 9 biomedical datasets — outperforms GPT-4, Med-PaLM-2 | Open |
| BioMedLM | Biomedicine | 2.7B | ~1.4 GB | Competitive with much larger models on PubMedQA/MedQA-USMLE | Open |
| SaulLM-7B/54B/141B | Legal | 7-141B | 3.5-70 GB | SOTA on LegalBench-Instruct. +6 pts over Mistral-Instruct | MIT |
| FinGPT | Finance | LoRA adapters | ~0.3 GB | F1 87.6% sentiment, 95.5% headline classification. $300/fine-tune | Open |
| GeoGalactica | Geoscience | 30B | ~15 GB | Galactica-based, adapted for geoscience | Open |

**Already in JARVIS Plan (Infrastructure Models)**

| Model | Role | Params | FP4 Size |
|-------|------|--------|----------|
| ThinkPRM | Verification | 1.5B | ~0.8 GB |
| R1-Distill-Qwen-1.5B | Speculative draft | 1.5B | ~0.8 GB |
| BERT classifier | Router | ~110M | ~0.06 GB |

### 11.4 Key Findings from Specialist Research

1. **Domain specialization consistently beats general models at smaller scale.** A fine-tuned 8B model outperformed GPT-3.5 by 46.5% on molecule generation (TOMG-Bench). BioMedLM at 2.7B rivals much larger general models on medical QA.
2. **The most impressive specialized models are NOT text LLMs.** ESM3, Evo 2, and GenCast operate on entirely different data modalities with specialized tokenization — proteins, DNA, atmospheric data.
3. **Continued pretraining + instruction tuning + alignment is the standard recipe** for text-based domain adaptation. SaulLM, BioMistral, ChemLLM, and FinGPT all follow this pattern — validating our physics/code brain training pipeline.
4. **LoRA/QLoRA makes domain adaptation accessible.** FinGPT demonstrated useful financial models for ~$300 via LoRA. This validates the JARVIS approach of LoRA adapters for HEP specialization.
5. **The LoRA adapter strategy scales deployment.** Adding new domain specializations (chemistry, biology, legal) within the same architecture family costs <1GB per adapter with millisecond swap times. New domains on different architectures require loading a separate base model (~16GB each, 5-10 seconds from SSD). JARVIS uses two bases (Qwen2.5 for physics/math, Qwen3 for code) — future specialists would attach to whichever base best fits their domain.

---

## 12. Local Deployment Hardware — NEW in v5

### 12.1 Primary Target: NVIDIA DGX Spark

| Spec | Value |
|------|-------|
| **Price** | $4,699 (Founders Edition, Feb 2026 pricing) |
| **Chip** | GB10 Grace Blackwell Superchip (ARM CPU + Blackwell GPU, fused via NVLink-C2C) |
| **Unified RAM** | 128 GB LPDDR5X |
| **Memory Bandwidth** | 273 GB/s |
| **AI Compute** | 1 PFLOP FP4 (sparse), ~100 TFLOPS FP16 |
| **Storage** | 4 TB NVMe SSD |
| **Networking** | 2× ConnectX-7 QSFP (200Gbps), 10GbE RJ-45, Wi-Fi |
| **Power** | 240W max (USB-C), ~100W typical AI workload |
| **OS** | DGX OS (Ubuntu-based), pre-installed NVIDIA AI stack |
| **Form Factor** | 5.9" × 5.9" × 2" (~2.6 lbs) |

**Key advantage over alternatives:** 128GB of CUDA-compatible unified memory at $4,699. No other NVIDIA product offers this memory capacity at this price. An RTX 5090 has 32GB VRAM ($2,500+), an RTX 6000 Pro has 96GB ($9,000+), and neither is a complete system.

**Key limitation:** Memory bandwidth (273 GB/s) is the bottleneck for token generation speed. For comparison, RTX 5090 delivers 1,792 GB/s (but only 32GB) and Mac Studio M3 Ultra delivers 819 GB/s (but no CUDA).

### 12.2 Memory Budget: What Fits

**Critical distinction: RAM vs storage.** To run a model, its weights must be loaded into RAM. The 128GB is the ceiling for what can be **actively running simultaneously**. The 4TB SSD stores model files on disk. Loading a 7B model from NVMe to RAM takes ~5-10 seconds; a 70B model takes ~30-60 seconds.

**Core JARVIS system (always resident in RAM):**

| Component | FP4 Size | Notes |
|-----------|----------|-------|
| R1-Distill-Qwen-32B (physics/math base) | ~16 GB | Qwen2.5 architecture — physics + math LoRA adapters |
| Qwen3-32B (code base) | ~16 GB | Qwen3 architecture — code LoRA adapters |
| Active LoRA adapter(s) | ~0.3-0.6 GB | Hot-swap within same base in milliseconds |
| Router classifier | ~0.1 GB | Always loaded |
| ThinkPRM verifier | ~0.8 GB | Always loaded |
| Draft model (speculative decoding) | ~0.8 GB | Always loaded |
| FAISS RAG index + overhead | ~5 GB | Always loaded |
| OS + framework overhead | ~10 GB | DGX OS + vLLM/TensorRT-LLM |
| **Subtotal (core system)** | **~50 GB** | |

**Remaining headroom: ~78 GB** — available for the 70B math brain and specialist models.

**⚠️ Two separate bases:** R1-Distill-Qwen-32B and Qwen3-32B are architecturally incompatible (different attention, tokenizers, layer structure). LoRA adapters trained on one CANNOT be loaded onto the other. Both are always resident. This costs ~32 GB instead of ~16 GB for a shared base, but preserves each brain's optimal starting architecture.

**Configuration A: Two bases + math LoRA (default)**

Math uses a LoRA adapter on the physics base (R1-Distill-Qwen-32B). Maximizes specialist headroom.

| Additional Component | FP4 Size |
|---------------------|----------|
| Math LoRA adapter (on physics base) | ~0.3 GB |
| ESM3-open (proteins) | ~0.7 GB |
| Evo 2 7B (genomics) | ~3.5 GB |
| ChemLLM-7B (chemistry) | ~3.5 GB |
| BioMistral-7B (biomedicine) | ~3.5 GB |
| **Total system** | **~62 GB** |
| **Remaining** | **~66 GB** |

All specialists loaded simultaneously. 66GB free for future models and KV cache.

**Configuration B: Two bases + separate 70B math brain**

| Additional Component | FP4 Size |
|---------------------|----------|
| R1-Distill-Llama-70B (math) | ~35 GB |
| Specialists loaded on demand from SSD | 0 (stored on 4TB SSD) |
| **Total system** | **~85 GB** |
| **Remaining** | **~43 GB** |

Math brain always resident alongside both 32B bases. Specialist models swap in/out from SSD in 5-10 seconds each. 43GB remaining is sufficient for ~12 specialist 7B models or KV cache headroom.

**What does NOT fit: R1-0528 (685B MoE)** requires ~342 GB at FP4. Exceeds even a 4-unit DGX Spark cluster (512GB when accounting for framework overhead). Math brain must use a distilled variant for local deployment.

### 12.3 Inference Speed Expectations

Based on published DGX Spark benchmarks (LMSYS, ServeTheHome, ProX PC):

| Model Size | Tokens/sec (decode) | Time to First Token | Use Case |
|-----------|--------------------|--------------------|----------|
| 7B (specialists) | ~40-50 tok/s | <1 sec | Snappy interactive |
| 20B | ~25-30 tok/s | ~2-5 sec | Good interactive |
| 32B (core brains) | ~10-15 tok/s | ~5-15 sec | Acceptable for GRACE |
| 70B (math brain) | ~4-5 tok/s | ~30-60 sec | Batch/async only |

For GRACE running HEP analysis tasks asynchronously (submit query → wait for answer → process), even the 70B math brain at 4-5 tok/s is workable. For rapid interactive use, the shared 32B base with LoRA swap is preferred.

### 12.4 Multi-Device Clustering

**Official: 2× DGX Spark via QSFP cable**

NVIDIA officially supports linking two Sparks via ConnectX-7 QSFP cable (200Gbps RDMA). Combined: 256GB unified memory, distributed inference via vLLM/TensorRT-LLM with NCCL. Up to 4 units can cluster (~512GB). Cost: ~$9,400-$18,800.

**Heterogeneous: DGX Spark + consumer GPU PC**

EXO Labs demonstrated DGX Spark + Mac Studio M3 Ultra over 10GbE, achieving **4× speedup** via disaggregated inference:
- DGX Spark handles compute-heavy **prefill** (processing the prompt)
- Second machine handles bandwidth-heavy **decode** (generating tokens)
- KV cache streams between them over the network

This works with any CUDA-capable machine. A spare PC with a 4060 Ti or 4070 could serve as:
- A dedicated host for lightweight specialist models (7B chemistry/bio/legal)
- A decode accelerator (if it has more memory bandwidth per GB than the Spark)
- Additional VRAM for tensor-parallel splits of larger models

**Requirements for heterogeneous clustering:**
- 10GbE minimum between machines (regular gigabit too slow for KV cache streaming)
- EXO framework (open source) or manual vLLM distributed setup
- Both machines must support the same model format (CUDA ecosystem)

**Cost-effective JARVIS cluster option:**
- DGX Spark (core brains): $4,699
- 10GbE NIC pair or switch: ~$100-200
- Existing PC with 4060 Ti/4070 (specialist model server): $0 (already owned)
- Total: ~$4,900

### 12.5 Deployment vs Previous Hardware Options

| Option | Memory | Bandwidth | AI Compute | Price | Status |
|--------|--------|-----------|------------|-------|--------|
| **DGX Spark (selected)** | 128 GB | 273 GB/s | 1 PFLOP FP4 | $4,699 | ✅ Best value for capacity |
| 2× DGX Spark cluster | 256 GB | 273 GB/s × 2 | 2 PFLOP FP4 | ~$9,400 | Fits 405B models |
| Mac Studio M3 Ultra 192GB | 192 GB | 819 GB/s | ~26 TFLOPS FP16 | ~$8,000 | Faster decode, no CUDA |
| Dual RTX 5090 | 64 GB | 1,792 GB/s | ~105 TFLOPS FP16 | ~$5,500+ | Fast but memory-limited |
| DGX Station (GB300) | 748 GB | TBD | 20 PFLOP | ~$50K-150K (est.) | Overkill — runs 1T param models |

The DGX Spark was selected as the primary deployment target because it offers the best ratio of memory capacity to cost in the NVIDIA ecosystem, comes with the full pre-installed AI software stack (CUDA, TensorRT-LLM, vLLM, Ollama, Docker), and supports official clustering for future expansion.

---

## 13. Key Papers (Updated)

### New additions from this iteration:
- **TTRL** (NeurIPS 2025): Test-Time Reinforcement Learning — Zuo et al. 211% AIME improvement with no labels
- **ETTRL** (2025): Entropy-based TTRL — halves compute, +5 points over standard TTRL
- **Absolute Zero Reasoner** (NeurIPS 2025 Spotlight): Self-play RL with zero external data — SOTA on math+code (tested up to 14B)
- **Heimdall** (2025): RL-trained math verifier, 94.5% accuracy. Pessimistic Verification: 54.2%→70% (16×) →83.3% (max) on AIME
- **LADDER** (2025): Self-guided curriculum via recursive problem simplification. 90% on MIT Integration Bee
- **PRIME** (Tsinghua, 2025): Process Reinforcement through Implicit Rewards — dense feedback from model's own probabilities
- **Guided by Gut** (2025): Self-guided test-time scaling without external PRM. 1.5B matches 32-70B performance
- **INTELLECT-2** (Prime Intellect, 2025): Distributed 32B RL training — validates GRPO at 32B scale on heterogeneous hardware
- **s1** (2025): Budget Forcing — SFT on 1K examples + "Wait" trick exceeded o1-preview on AIME by 27%
- **S\*** (NovaSky AI, 2025): Sequential + parallel test-time scaling with distinguishing inputs — R1-Distill-Qwen-32B → 85.7% LiveCodeBench
- **NTele-R1-32B** (2025): Data-Efficient Distillation — 0.8K curated examples → 81.87% AIME 2024, validates quality-over-quantity

### New additions in v5 (specialist ecosystem & deployment):
- **ESM3** (EvolutionaryScale, Science 2025): 98B protein foundation model. ESM3-open (1.4B) publicly available. Generated novel GFP equivalent to 500M years of evolution
- **Evo 2** (Arc Institute, Nature 2026): 40B DNA foundation model. 9.3T nucleotides from all domains of life. Generated a functional bacteriophage. Fully open source
- **ChemLLM** (AI4Chem, 2024): First chemistry-dedicated LLM with ChemBench evaluation suite. GPT-4 competitive
- **BioMistral** (ACL 2024 Findings): Mistral 7B continually pre-trained on PubMed Central. First large-scale multilingual medical LLM evaluation
- **OpenBioLLM-70B** (2024): LLaMA-3-70B + DPO fine-tuning. 86.06% across 9 biomedical datasets — outperforms GPT-4
- **SaulLM** (NeurIPS 2024): Legal LLMs from 7B to 141B (Mixtral-based). 400B+ legal tokens. MIT License
- **FinGPT** (AI4Finance, 2023–2025): Open-source financial LLMs via LoRA. ~$300 per fine-tune vs BloombergGPT's ~$2.7M
- **EXO Labs** (2025): Disaggregated inference across heterogeneous hardware. DGX Spark + Mac Studio → 4× speedup
- **GenCast** (DeepMind, Nature 2024): Diffusion-based ensemble weather model. Outperforms ECMWF ENS on 97.2% of targets. Open source
- **KVQuant** (UC Berkeley, 2024): Sub-4-bit KV cache quantization enabling 10M+ token context. Per-channel key quantization + pre-RoPE + non-uniform datatypes
- **AQUA-KV** (ICLR 2025): Adaptive KV cache quantization using cross-layer prediction. Near-lossless at 2-2.5 bits per value, <1% perplexity degradation on Llama 3.2

### Retained from previous iterations:
- DeepSeek-R1 / R1-0528, RouteLLM, DoRA, QLoRA, POME, ThinkPRM, Snell et al. (test-time compute scaling), DARE/TIES merging, DeepSWE, SWE-RL, Light-R1-32B, Sakana evolutionary merging, FW-Merging

---

## 14. Errata Log

All corrections from previous iterations are retained (see Appendix C of franken_model_system_research.md). Additional corrections in this version:

12. **TTRL AIME improvement figure:** The paper reports both 159% (v1) and 211% (v3) depending on version. The 211% figure is from the updated v3 paper (June 2025). Both are for Qwen-2.5-Math-7B on AIME 2024.
13. **TTRL hardware requirement:** Published experiments used 8× A100 **80GB** GPUs. Delta has A100 **40GB**. For 32B model TTRL/ETTRL, H200 nodes are required (gradient updates during inference need more memory than pure inference). Budgeted accordingly at 3× SU rate.
14. **Absolute Zero Reasoner scope:** AZR uses code execution as the verifier, making it directly applicable to code reasoning but NOT directly applicable to physics multiple-choice reasoning (GPQA). Physics brain uses standard GRPO instead.
15. **R1-Distill-Qwen-32B GPQA score:** Verified as 62.1% (from DataCamp/DeepSeek sources), not the 60% estimated in earlier drafts.

**New in v4 (audit findings):**

16. **AZR tested only up to 14B:** The paper evaluates Qwen-Coder 3B, 7B, and 14B. 32B is extrapolation based on the scaling trend (+5.7, +10.2, +13.2 OOD gains). The veRL framework supports 32B, but AZR's propose-and-solve dynamics are unverified at this scale. Mitigation: 200 SU allocated for early validation; GRPO fallback if needed.
17. **AZR algorithm is REINFORCE++, not GRPO:** Both run on veRL, but the advantage estimator and multi-task structure differ. Hyperparameters from GRPO papers don't transfer directly.
18. **Heimdall 83.3% requires N=64, M=256:** This is 16,384 model calls per problem — impractical for interactive use. The practical deployment target is the 16× tier (N=16 + verification → 70% on AIME). Plan now distinguishes practical vs batch ceiling throughout.
19. **Heimdall validated on math only:** Transfer to physics (GPQA) and code (LiveCodeBench) uncertain. ThinkPRM shows +8% on GPQA subset and +4.5% on LiveCodeBench — smaller than Heimdall's +16-29% on AIME. Physics GPQA projection revised from 82-87% to 78-84%.
20. **ETTRL budget was tight:** 600 SU at 3× rate = 200 H200 GPU-hrs. For 500 problems × 60 episodes × 64 samples on a 32B model, this risks running out with gradient checkpointing overhead. Increased to 900 SU (300 H200 GPU-hrs).
21. **Budget Forcing was missing:** The s1 paper showed SFT on 1K examples + appending "Wait" to force continued thinking exceeded o1-preview on AIME by 27%. Zero-cost inference technique now added.
22. **S\* was missing:** Pushes R1-Distill-Qwen-32B to 85.7% on LiveCodeBench via sequential+parallel scaling with execution-based distinguishing inputs. Now added to code brain inference pipeline.
23. **NTele-R1-32B validates data strategy:** Achieved 81.87% on AIME 2024 with only 0.8K curated examples. Confirms that extremely small, high-quality distillation datasets can produce massive gains — supporting our emphasis on quality filtering over quantity.
24. **ThinkPRM verification gain revised:** Previously cited as +5-15% uniformly. Now domain-specific: +5-15% on math, +3-8% on physics, +4-8% on code. The variance reflects the different difficulty of verification across domains.

**New in v5:**

25. **JARVIS is not an MoE:** Clarified that JARVIS is a routed ensemble / specialist system, not a true Mixture of Experts. True MoE (DeepSeek-V3, Mixtral) is a single model with shared attention and token-level routing trained end-to-end. JARVIS uses completely separate models with query-level external routing.
26. **R1-0528 cannot deploy locally:** At 685B parameters (even MoE with ~37B active), FP4 size is ~342GB — exceeds DGX Spark's 128GB and even a 2-unit cluster's 256GB. Math brain defaults to R1-Distill-Llama-70B (~35GB FP4) or R1-Distill-Qwen-32B (~16GB FP4) for local deployment.
27. **DGX Spark memory bandwidth is the bottleneck:** 273 GB/s LPDDR5X limits decode speed to ~4-5 tok/s for 70B models and ~10-15 tok/s for 32B models. CES 2026 software update delivered 2.5× improvement via TensorRT-LLM optimizations. GRACE's async query pattern makes this acceptable.
28. **DGX Spark price increased:** MSRP raised from $3,999 to $4,699 in February 2026 due to global memory supply constraints. Hardware specs unchanged.
29. **Specialist models are stored on SSD, loaded to RAM on demand:** The 4TB NVMe can store 500+ specialist models at FP4. Only active models occupy the 128GB RAM budget. Swap time is 5-10 seconds for 7B models.
30. **Context window limited by KV cache, not just architecture:** R1-Distill-Qwen-32B architecturally supports 128K (Qwen2.5 with YaRN), but DeepSeek only validated up to 32K. Without KV cache optimization, parallel best-of-16 at 32K costs ~64GB of KV cache (FP16), capping practical context. With FP8 KV (vLLM default flag), this halves. With 2-bit KVQuant, parallel best-of-16 at 64K becomes feasible. SSD offload enables 128K+ for single-stream derivations.
31. **"128K physics derivations" was misleading:** The v4 claim of "up to 128K" for physics derivations was only achievable with single-pass inference. With best-of-N amplification, the practical ceiling was 32K without KV cache optimization. Now corrected: 128K is achievable with FP8 KV + SSD offload for single pass, or with 2-bit KVQuant for parallel best-of-16 at 64K.
32. **KV cache optimization techniques added to free techniques table:** FP8 KV cache, 2-bit KVQuant, SSD offload, and context compression added as zero-cost inference techniques. These collectively extend JARVIS's effective context by 6-8× with negligible quality impact.
33. **Two-base architecture resolved:** Physics brain uses R1-Distill-Qwen-32B (Qwen2.5) and code brain uses Qwen3-32B (Qwen3). Architecturally incompatible — adapters cannot be shared across bases. Both bases always resident (~32 GB total at FP4 vs ~16 GB for a hypothetical shared base). Core system footprint increased from ~34 GB to ~50 GB. Config A available headroom: ~66 GB. Config B available headroom: ~43 GB. Both sufficient for deployment.
