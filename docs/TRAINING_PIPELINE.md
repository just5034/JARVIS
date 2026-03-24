# JARVIS Training Pipeline

**Platform:** NCSA Delta via ACCESS-CI allocation
**Budget:** 8,000 Service Units (SUs)
**Hardware:** A100 40GB GPUs (1 SU/GPU-hr) + H200 141GB GPUs (3 SU/GPU-hr)

---

## Budget Allocation

| Phase | SUs | GPU-hrs | System | Brain |
|-------|-----|---------|--------|-------|
| Data generation | 350 | 350 A100-hrs | A100 | Physics |
| Distillation SFT | 800 | 800 A100-hrs | A100 | Physics |
| Curriculum GRPO | 2,000 | 2,000 A100-hrs | A100 | Physics |
| ETTRL polish | 900 | 300 H200-hrs | H200 (3×) | Physics |
| Self-distillation | 450 | 450 A100-hrs | A100 | Physics |
| AZR self-play RL | 2,000 | 2,000 A100-hrs | A100 | Code |
| Targeted SFT | 300 | 300 A100-hrs | A100 | Code |
| Self-distillation | 300 | 300 A100-hrs | A100 | Code |
| Router + eval | 200 | 200 A100-hrs | A100 | Router |
| Exploration buffer | 700 | 700 A100-hrs | A100 | Contingency |
| **Total** | **8,000** | | | |

---

## Physics Brain Training

### Base Model
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- **Baseline:** 62.1% GPQA Diamond
- **Target:** 78-84% GPQA Diamond after all phases

### Phase A: Data Generation (~350 SU)

**A1. Multi-teacher traces (~200 SU)**
- Run R1-0528 (via API or quantized) on 50K physics/chemistry/biology problems
- Generate 8 traces per problem (400K raw traces)
- Sources: GPQA train split, SciInstruct, graduate textbook problems, Olympiad physics, ARC-Challenge, arXiv problem sets
- Use DeepSpeed inference for efficient batch generation

**A2. LADDER curriculum generation (~50 SU)**
- For hardest 5K problems, recursively generate easier variants using the teacher
- Creates difficulty ladder: easy → medium → hard for each problem family
- Output: ~15K additional curriculum problems

**A3. Rejection sampling + quality filtering (~50 SU)**
- From 400K raw traces, filter to ~100K high-quality examples
- Criteria: answer correctness (verify against ground truth), LLM judge for reasoning quality (AskLLM), diversity filtering, conciseness preference

**A4. Synthetic textbook chapters (~50 SU)**
- Generate ~5K textbook-style explanations from R1-0528
- Covers core physics, chemistry, biology concepts
- Teaches conceptual understanding, not just problem patterns

### Phase B: Distillation SFT (~800 SU)

**Setup:**
- Fine-tune on 4× A100 40GB with DeepSpeed ZeRO-3
- Method: QDoRA (quantized DoRA, rank=32)
- Base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

**Data mix:**
- 70% problem-solving traces (70K)
- 20% textbook explanations (20K)
- 10% general instruction data (10K) — prevents catastrophic forgetting

**Hyperparameters (starting point):**
```yaml
learning_rate: 2e-5
warmup_ratio: 0.03
num_epochs: 3
batch_size: 4 (per GPU)
gradient_accumulation_steps: 4
max_seq_length: 8192
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
use_dora: true
```

**Evaluation checkpoint:** Run GPQA Diamond eval after each epoch. Target: 68-72%.

### Phase C: Curriculum GRPO RL (~2,000 SU)

**Setup:**
- Framework: veRL (same as AZR, shared infrastructure)
- 4× A100 40GB per run with DeepSpeed ZeRO-3

**Curriculum stages:**
1. Steps 1-100: Undergrad-level physics problems
2. Steps 100-300: Graduate-level problems
3. Steps 300-500+: Competition/research-level problems

**Reward signals:**
- Numerical answer verification (binary: correct/incorrect)
- Unit/dimensional check (+0.1 for correct units)
- Reasoning structure reward (penalize skipped steps)
- Length penalty (prefer concise correct solutions)

**PRIME implicit rewards:**
- Extract process-level reward signals from reference model log-probabilities
- Zero additional compute — uses the frozen reference model already required by GRPO

**Self-refinement:**
- Two-pass generation per problem: solve → critique → correct
- Reward applied to corrected answer
- ~2× generation cost per problem but trains internal error-correction

**Evaluation:** GPQA Diamond after every 100 steps. Target: 72-78%.

### Phase D: ETTRL Polish (~900 SU on H200)

**Why H200:** GRPO weight updates during inference require full model + optimizer states in memory — exceeds 40GB on A100 for a 32B model. H200 has 141GB.

**Setup:**
- 500 hard physics problems without ground-truth labels
- 64 solutions per problem, majority voting as pseudo-labels
- ETTRL branches only at high-entropy tokens (halves compute vs standard TTRL)
- 40-60 episodes

**Budget note:** 900 SU at 3× rate = 300 H200 GPU-hrs. Tight but sufficient with gradient checkpointing.

**Target:** +2-4% on hard physics problems.

### Phase E: Post-Processing (450 SU for self-distillation; rest is free)

1. **POME (0 SU):** SVD projection on weight deltas. Run on CPU. +1-2.5%.
2. **Checkpoint SLERP merging (0 SU):** Merge best 3-5 RL checkpoints. CPU. +0.5-1%.
3. **Self-distillation (450 SU):** Take model's best outputs on training set → filter for quality → one additional SFT pass. +1-3%.

### Final evaluation: GPQA Diamond. Target: 78-84%.

---

## Code Brain Training

### Base Model
- **Model:** `Qwen/Qwen3-32B`
- **Baseline:** ~45% LiveCodeBench
- **Target:** 65-72% LiveCodeBench after all phases

### Phase F: Absolute Zero Self-Play RL (~2,000 SU)

**⚠️ Risk:** AZR tested only up to 14B. 32B is extrapolation. Budget 200 SU from exploration buffer for early validation.

**Early validation (200 SU):**
- Run AZR on Qwen3-32B for 50 steps
- Check for: stable training loss, sensible proposed problems, mode collapse in proposer
- If issues → abort AZR, fall back to standard GRPO with curated code datasets

**Full AZR training (if validation passes):**
- Model simultaneously proposes coding challenges and solves them
- Python executor validates both proposed tasks and solutions
- Three reasoning types: abduction, deduction, induction
- Algorithm: REINFORCE++ (not GRPO) with multi-task advantage estimator
- Framework: veRL
- Seed: single identity-function triplet
- Train for ~1,800 A100 GPU-hours on 4× A100 nodes

**GRPO fallback (if AZR fails at 32B):**
- Standard GRPO with curated code datasets (LiveCodeBench, CodeContests, APPS)
- Same veRL framework, just switch algorithm and data source
- Budget: same 2,000 SU

### Phase G: Targeted SFT (~300 SU)

- QDoRA fine-tuning on LiveCodeBench problems with R1-0528 solutions (rejection sampled)
- Competition programming problems (CodeContests, APPS) with verified solutions
- ~30K high-quality examples

### Phase H: Post-Processing (~300 SU for self-distillation; rest free)

1. POME projection (0 SU)
2. Checkpoint merging (0 SU)
3. One self-distillation cycle (300 SU)

### Final evaluation: LiveCodeBench. Target: 65-72%.

---

## Router Training (~200 SU)

**After core brains are trained:**

1. Run each brain on a validation set of ~5K problems per domain
2. Label difficulty: correct in 1 pass = easy, correct in best-of-4 = medium, otherwise = hard
3. Fine-tune two `bert-base-uncased` classifiers:
   - Domain classifier: math / physics / code (+ specialist domains)
   - Difficulty classifier: easy / medium / hard
4. Evaluate routing accuracy on held-out set

---

## Export and Deployment

After all training phases:

1. Export LoRA adapters for physics brain (general + HEP)
2. Export LoRA adapters for code brain (general + HEP)
3. Export router classifier weights
4. Apply POME + checkpoint merging (CPU, no SU cost)
5. Quantize to NVFP4 for DGX Spark deployment
6. Transfer all artifacts to DGX Spark
7. Verify benchmark scores match Delta results post-quantization

---

## Evaluation Benchmarks

| Benchmark | Brain | How to Run |
|-----------|-------|-----------|
| GPQA Diamond | Physics | 198 graduate-level science MCQ, 0-shot |
| AIME 2024 | Math | 30 competition math problems, score by correct/total |
| AIME 2025 | Math | 30 problems, newer and harder |
| LiveCodeBench | Code | Coding problems with execution-based verification |
| MATH-500 | Math | 500 math problems, accuracy metric |
| HumanEval | Code | Function completion, pass@1 |

**Evaluation cadence:** After every major training phase, run the relevant benchmark. Log all results with timestamps for tracking progress.
