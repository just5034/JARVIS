# JARVIS Master Brainstorm Document

**Created:** 2026-04-04
**Purpose:** Ongoing strategic brainstorming for JARVIS as general-purpose AI infrastructure

---

## Vision Statement

JARVIS is not just a GRACE backend — it is a **general-purpose self-hosted LLM inference system** designed to plug into any agentic framework. GRACE is one client (and the academic justification), but the real goal is a personal AI infrastructure layer with reasoning capabilities competitive enough to replace frontier API calls across arbitrary workloads: coding agents, research agents, general-purpose assistants, and any future framework.

---

## 1. Competitive Landscape (April 2026)

### Qwen3.5-27B vs Frontier

| Benchmark | Qwen3.5-27B | GPT-5.2 | Claude Opus 4.6 | Gemma 4 | Gap |
|-----------|------------|---------|-----------------|---------|-----|
| GPQA Diamond | 86% | ~90%+ | ~93% | 84% | ~7% behind top |
| AIME 2026 | 81% | 96.7% | 93.3% | 89% | ~12-15% behind |
| LiveCodeBench | 80.7% | ~85%+ | ~85%+ | 80% | ~5% behind |
| SWE-bench Verified | 72.4% | ~75%+ | ~78%+ | — | ~5% behind |
| IFBench (instruction) | 76.5% | 75.4% | 58.0% | — | **Beats frontier** |
| BFCL-V4 (tool use, 122B variant) | 72.2% | 55.5% (mini) | — | — | **Beats frontier** |

### Qwen3.5-27B vs Open-Source <30B

- Gemma 3 27B: crushed (43-point GPQA gap, 51-point LiveCodeBench gap)
- Nemotron Nano 30B: competitive on knowledge, weaker on reasoning
- Qwen3-Coder-30B-A3B (MoE): faster but lower quality
- **Qwen3.5-27B is the clear king of the <30B dense class**

### Key Strengths

- Instruction following / tool use: actually **beats** frontier models
- Agentic coding: ties GPT-5 mini on SWE-bench (72.4%)
- 262K native context (1M extended)
- 14 GB at FP4 — fits easily on DGX Spark with ~90 GB headroom
- Apache 2.0 license

### Key Weaknesses

- Hard math/reasoning: 81% AIME vs 93-97% frontier (biggest gap)
- Complex multi-step reasoning chains: where single-pass inference hits limits
- No native HEP domain knowledge (fixable with LoRA)

---

## 2. Levers for Closing the Reasoning Gap

### Lever 1: Inference-Time Compute (Already Built)

| Technique | Expected Gain | Status |
|-----------|--------------|--------|
| Best-of-16 + self-consistency | +5-10% on math/reasoning | Built |
| ThinkPRM verification (pessimistic selection) | +4-8% over naive voting | Built |
| Budget forcing ("Wait" trick, 3x) | +2-5% on hard problems | Built |
| S* code verification (execute + distinguish) | +5-10% on code tasks | Built |
| Verification chains (substitution/dimensional checks) | +2-3% | Built |

**Stacked: plausibly 81% → 90%+ AIME, 80% → 88%+ LiveCodeBench.**

Research backing: "scaling inference compute with inference strategies can be more computationally efficient than scaling model parameters" (Berkeley, 2024).

### Lever 2: GRPO Reinforcement Learning (~2,000-3,000 SU)

- GRPO on Qwen models shows consistent math/reasoning improvements
- PTA-GRPO (planning-guided) shows "stable and significant improvements"
- S-GRPO: +0.7-6% accuracy while cutting output length 35-60%
- Community Qwen3.5 fine-tunes: clear gains on agentic coding, mixed on general benchmarks
- **Best targets:** agentic tool use, multi-step reasoning, code generation + self-correction

### Lever 3: Better Verifiers

- **VPRMs** (Verifiable Process Reward Models): +20% F1 over standard PRMs
- **CompassVerifier**: lightweight, multi-domain
- **ThinkPRM**: still top-tier — beats PRM800K discriminative verifiers by 8% OOD
- Worth evaluating whether verifier upgrade gives more ROI than GRPO

### Lever 4: Domain LoRA Adapters

- HEP Physics LoRA (~200 SU): particle physics, detector design
- HEP Code LoRA (~200 SU): Geant4, ROOT, Pythia8 patterns
- Tool-calling LoRA (~200 SU): better function calling across all agents
- **Important:** Qwen3.5 needs its own tool-calling template (not ReAct) — stop words can appear inside reasoning

### Lever 5: Multi-Pass Reasoning Architecture (NEW — see Section 4)

---

## 3. Revised Priority Stack

| Priority | Action | SU Cost | Impact |
|----------|--------|---------|--------|
| 1 | Confirm baseline evals (running now) | ~20 | Decision gate |
| 2 | Benchmark inference amplification stack on Delta | ~50 | Measure actual stacked gains |
| 3 | GRPO on agentic tasks (tool use, multi-step, code) | 2,000-3,000 | Push base reasoning |
| 4 | Evaluate upgraded verifier (VPRM / CompassVerifier) | ~50 | Better selection = free accuracy |
| 5 | HEP LoRA adapters (for GRACE) | ~400 | Domain specialization |
| 6 | Tool-calling LoRA / template optimization | ~200 | Better function calling |
| 7 | Router retrain for difficulty calibration | ~50 | Right-size compute |

**Budget:** ~7,924 SU remaining. Total planned: ~3,000-3,800 SU. Buffer: ~4,000+ SU.

---

## 4. Multi-Pass Reasoning Architecture (Core Innovation Idea)

### The Problem

Current LLM inference — even with best-of-N and verification — treats each generation as an **independent, monolithic pass**. The model gets one shot to produce a complete chain of thought. For hard problems, this is fundamentally limiting:

- Long reasoning chains accumulate errors (each step has some probability of going wrong)
- The model can't "step back" and reconsider earlier assumptions mid-generation
- No way to build up partial understanding incrementally
- No persistence of intermediate insights across attempts

Humans don't solve hard problems this way. We:
1. Break the problem into subproblems
2. Solve pieces, write down intermediate results
3. Revisit and revise earlier conclusions when later steps reveal issues
4. Build a "scratchpad" of verified facts that accumulates over time
5. Try multiple approaches and synthesize insights across attempts

### The Proposal: Reasoning Memoization / Caching Framework

**Instead of N independent passes, use a multi-pass architecture where each pass builds on verified intermediate results from previous passes.**

#### Key Components (to research and design):

**a) Reasoning Decomposition**
- Automatically break complex queries into sub-problems
- Each sub-problem gets its own focused generation pass
- Results are verified independently before being composed

**b) Reasoning Cache / Memo**
- Persistent scratchpad of verified intermediate results
- Subsequent passes can READ from the cache (proven facts, partial solutions, failed approaches)
- Cache entries are tagged with confidence scores from the verifier

**c) Iterative Refinement Passes**
- Pass 1: Initial attempt — generate full reasoning chain
- Pass 2: Verify each step, identify weak links, cache strong intermediate results
- Pass 3: Re-attempt with cached facts injected as "known truths"
- Pass N: Continue until verifier confidence exceeds threshold or budget exhausted

**d) Cross-Attempt Synthesis**
- Don't just vote on final answers — extract and merge insights from across attempts
- If attempt A gets steps 1-3 right but fails at step 4, and attempt B gets steps 3-5 right, synthesize the best path

**e) Failure Memory**
- Track what approaches DIDN'T work and why
- Inject "do not repeat" signals into subsequent passes
- Analogous to how humans learn from failed attempts within a single problem-solving session

### Why This Could Be a Step Change

Current best-of-N generates 16 independent chains and picks the best. This is wasteful — each chain learns nothing from the others. A multi-pass architecture with memoization could:

1. **Achieve best-of-N quality with fewer total tokens** (each pass is informed by prior work)
2. **Handle problems that NO single pass can solve** (problems requiring iterative refinement)
3. **Provide interpretable reasoning traces** (the cache shows exactly what was verified and when)
4. **Scale reasoning depth, not just breadth** — current best-of-N scales breadth (more independent attempts); multi-pass scales depth (deeper, more refined reasoning)

### Research Questions

- What decomposition strategies work best? (LLM-driven vs rule-based vs hybrid)
- How to represent the reasoning cache? (structured facts vs natural language vs embeddings)
- What's the optimal pass schedule? (fixed N passes vs adaptive based on verifier confidence)
- How does this interact with existing best-of-N? (can you do multi-pass AND best-of-N?)
- Memory overhead: can the cache fit in KV context or does it need external storage?
- How to handle contradictions between passes?

### Prior Art Landscape (Researched 2026-04-04)

What exists, what each does well, and **what's still missing** (i.e., where our idea lives):

#### Tier 1: Single-Pass Structured Reasoning

| System | What It Does | Limitation |
|--------|-------------|------------|
| **Chain-of-Thought** | Linear step-by-step reasoning | No backtracking, no verification, one shot |
| **Tree-of-Thought (ToT)** | BFS/DFS over branching thought steps, LLM evaluates each node | Still single-session — no memory across retries. Tree structure is rigid. |
| **Graph-of-Thought (GoT)** | Arbitrary DAG of thoughts, can merge/refine nodes | More flexible than ToT (+62% on sorting, -31% cost), but still no persistent cross-attempt memory |

**Gap:** These decompose within a single generation session. They don't accumulate verified knowledge across multiple attempts.

#### Tier 2: Iterative Self-Refinement

| System | What It Does | Limitation |
|--------|-------------|------------|
| **Self-Refine** | Generate → self-critique → refine, repeat | No external verification — model critiques itself (can reinforce errors). No decomposition. |
| **Reflexion** (NeurIPS 2023) | Generate → evaluate → verbal self-reflection stored in memory → retry | **Closest to our idea.** Has episodic memory. But: reflections are vague ("I should try harder"), not verified intermediate facts. Memory is unstructured text. +22% on AlfWorld, +20% HotPotQA, +11% HumanEval. |

**Gap:** Self-Refine has no memory. Reflexion has memory but it stores *reflections about failures*, not *verified partial results*. Neither decomposes the problem into sub-steps that are independently verified and cached.

#### Tier 3: Search-Based Reasoning

| System | What It Does | Limitation |
|--------|-------------|------------|
| **LATS** (ICML 2024) | Monte Carlo Tree Search over LLM actions. LLM-powered value function + self-reflections. External environment feedback. | Powerful but heavy — requires many LLM calls for node expansion + evaluation. Designed for agent actions (web nav, coding), not pure reasoning. |
| **SC-MCTS*** | MCTS for reasoning, outperformed o1-mini by 17.4% on Blocksworld with Llama-3.1-70B | Promising results. But: high compute cost, complex to implement, no persistent cross-problem memory. |
| **SWE-Search** | MCTS + self-improvement for code agents. +23% over non-MCTS agents. | Code-specific. Shows MCTS works for agentic tasks. |
| **Empirical-MCTS** | "Remembering past reasoning patterns" — SOTA on AIME25, ARC-AGI-2 | **Key insight: memory of past reasoning IS the breakthrough.** Closest to our memoization idea but focused on cross-problem transfer, not within-problem caching. |

**Gap:** MCTS approaches are powerful but computationally expensive and don't efficiently cache/reuse verified intermediate results within a single problem-solving session.

#### Tier 4: Hybrid Frameworks

| System | What It Does | Limitation |
|--------|-------------|------------|
| **ReTreVal** (Jan 2026) | Tree-of-Thoughts + self-refinement + critique scoring + reflexion memory buffer. Adaptive depth. Top-k pruning. Cross-problem learning. | **Most complete existing system.** Tested on Qwen 2.5 7B, outperforms ReAct/Reflexion/Self-Refine on math + creative tasks. BUT: the "reflexion memory" is still about patterns, not verified intermediate computations. |
| **Cache Saver** (EMNLP 2025) | Modular caching framework for multi-step reasoning. Saves 60% cost. | Focused on KV cache efficiency, not semantic caching of reasoning results. |

**Gap:** ReTreVal is the closest existing work. What it's missing: (1) a **verified fact cache** where intermediate results are independently checked and promoted to "known truths," (2) **cross-attempt result synthesis** (not just pattern memory), and (3) **decomposition-aware caching** where sub-problem solutions are reused.

#### The Cognitive Science Angle

Human working memory research confirms the intuition:
- Working memory capacity is ~7 chunks — humans compensate by **chunking** (compressing verified sub-results into single units)
- Experts solve problems with **less** working memory load because they've pre-chunked domain knowledge
- LLMs lack working memory entirely — they have a context window but no active manipulation/compression mechanism
- Key finding: "LLMs struggle with novel and constrained problems, indicating limitations in their ability to generalize beyond training data" — exactly the problems where iterative refinement matters most

**The analogy:** Our reasoning cache IS the working memory that LLMs lack. Verified intermediate results are "chunks." The multi-pass architecture is the iterative refinement loop that humans use naturally.

### What's Actually Novel in Our Idea

After surveying the field, here's what doesn't exist yet:

1. **Verified Fact Cache (not reflection memory):** Reflexion/ReTreVal store "what went wrong" and "patterns." We want to store "what is PROVEN TRUE so far" — intermediate results that passed verification. This is a fundamentally different kind of memory.

2. **Decompose → Solve → Verify → Cache → Compose pipeline:** No existing system does all five steps in a loop. ToT decomposes but doesn't verify/cache. Reflexion verifies but doesn't decompose. Cache Saver caches but at the KV level, not the semantic level.

3. **Cross-attempt result merging:** Best-of-N picks ONE winner. We want to take step 1-3 from attempt A and step 4-6 from attempt B and compose a solution that neither attempt produced alone.

4. **Depth scaling (not just breadth scaling):** Best-of-N scales breadth (more independent attempts). MCTS scales search. Our approach scales DEPTH — each pass goes deeper because it starts from verified ground, not from scratch.

5. **Failure memoization with specificity:** Reflexion's reflections are vague ("I made an arithmetic error"). Our failure cache would store "step 3 yielded X=42 which contradicts constraint Y" — specific enough to prevent the exact same error.

### Proposed Architecture: ARIA (Adaptive Reasoning with Iterative Accumulation)

Working name. The system sits as a layer BETWEEN the JARVIS router/inference engine and the final response — it orchestrates multiple passes through the model with persistent state.

```
Query (hard) ──▶ JARVIS Router ──▶ difficulty = hard ──▶ ARIA Engine
                                                            │
                                                    ┌───────▼────────┐
                                                    │  DECOMPOSER    │
                                                    │  Break into    │
                                                    │  sub-problems  │
                                                    └───────┬────────┘
                                                            │
                              ┌──────────────────────────── │ ◀── Reasoning Cache
                              │                             │      (verified facts,
                              ▼                             ▼       failed approaches)
                     ┌────────────────┐           ┌────────────────┐
                     │  SOLVE pass    │           │  SOLVE pass    │
                     │  sub-problem A │           │  sub-problem B │
                     └───────┬────────┘           └───────┬────────┘
                             │                            │
                     ┌───────▼────────┐           ┌───────▼────────┐
                     │  VERIFY        │           │  VERIFY        │
                     │  (ThinkPRM /   │           │  (ThinkPRM /   │
                     │   VPRM /       │           │   execution)   │
                     │   execution)   │           │                │
                     └───────┬────────┘           └───────┬────────┘
                             │                            │
                     ┌───────▼────────┐           ┌───────▼────────┐
                     │  CACHE result  │           │  CACHE result  │
                     │  if verified   │           │  if verified   │
                     │  (or cache     │           │                │
                     │  failure mode) │           │                │
                     └───────┬────────┘           └───────┬────────┘
                             │                            │
                             └──────────┬─────────────────┘
                                        │
                                ┌───────▼────────┐
                                │  COMPOSE       │
                                │  Merge verified │
                                │  sub-results   │
                                │  into answer   │
                                └───────┬────────┘
                                        │
                                ┌───────▼────────┐
                                │  META-VERIFY   │
                                │  Check composed│
                                │  answer as     │
                                │  whole         │
                                └───────┬────────┘
                                        │
                              ┌─────────▼──────────┐
                              │ Confidence ≥ θ ?   │
                              │                    │
                              │  YES → return      │
                              │  NO  → next pass   │
                              │  (inject cache     │
                              │   into context)    │
                              └────────────────────┘
```

#### Component Details

**1. Decomposer**
- LLM-driven (not rule-based) — ask the model to break the problem into sub-steps
- For math: "What intermediate results do I need?"
- For code: "What functions/components do I need to build?"
- For reasoning: "What facts do I need to establish?"
- Adaptive granularity: harder problems → finer decomposition

**2. Reasoning Cache (the core innovation)**
```
Cache Entry:
{
  "sub_problem": "Compute the integral of x²sin(x) from 0 to π",
  "result": "π² - 4",
  "confidence": 0.95,        # From verifier
  "method": "integration by parts, applied twice",
  "verification": "checked by substitution",
  "pass_number": 2,
  "attempts_failed": [
    {"result": "π² - 2", "reason": "sign error in second IBP step"}
  ]
}
```
- Stored as structured JSON, injected into prompt as "established facts" for subsequent passes
- Entries have confidence scores from the verifier
- Failed attempts stored with specific failure reasons
- Cache persists across passes within a single query (not across queries — that's a future extension)

**3. Multi-Pass Schedule**
- **Pass 1:** Full attempt, no cache. Decompose problem. Verify each step. Cache high-confidence intermediate results + specific failure modes.
- **Pass 2:** Re-attempt with cache injected. "The following facts have been verified: [cache]. The following approaches failed: [failures]. Solve the remaining sub-problems."
- **Pass 3+:** Continue until (a) meta-verifier confidence ≥ threshold, (b) no new cache entries added (convergence), or (c) compute budget exhausted.
- **Adaptive:** Easy problems exit after pass 1. Only hard problems get multiple passes.

**4. Cross-Attempt Synthesis**
- After N passes, don't just take the last answer
- Extract the highest-confidence sub-result for each sub-problem across ALL passes
- Ask the model to compose a final answer from these best pieces
- This is the "take steps 1-3 from attempt A, step 4 from attempt B" operation

**5. Integration with Existing JARVIS Pipeline**
- ARIA sits inside the `hard` difficulty strategy in `inference.yaml`
- For easy/medium queries: existing single-pass / best-of-N pipeline (unchanged)
- For hard queries: ARIA replaces or wraps the current best-of-16 approach
- Can combine: run best-of-4 ARIA passes instead of best-of-16 independent passes (4 informed passes vs 16 blind ones)

#### Expected Gains (Hypotheses to Test)

| Metric | Current (best-of-16) | ARIA (4 informed passes) | Why |
|--------|---------------------|-------------------------|-----|
| AIME accuracy | ~88% (estimated w/ amplification) | 92-95%? | Iterative refinement catches errors that voting misses |
| Token efficiency | 16x base cost | 6-8x base cost? | Each pass is shorter (cache provides "known truths") |
| Problems solvable | Limited by single-pass ceiling | Higher ceiling | Can solve problems that require iterative insight |
| Latency | ~30-60s (parallel gen) | ~60-120s (sequential passes) | Tradeoff: slower but more accurate |

#### Minimum Viable Prototype

Start simple. Test on AIME problems:
1. **Pass 1:** Generate solution with CoT. Verify each step with ThinkPRM. Cache verified steps.
2. **Pass 2:** Inject verified steps + failure notes. "Given that [verified facts], and avoiding [failed approaches], solve the problem."
3. **Compare:** 2-pass ARIA vs 2x independent best-of-2 vs single best-of-4.

If 2-pass beats best-of-4 on AIME, the architecture is validated and worth scaling up.

---

## 5. DGX Spark Deployment Advantages

- **$4,699 one-time** vs $500-5,000/month API costs
- **15-20 tok/s** on 27B at FP4 — acceptable for agent workloads
- **128 GB unified RAM** — room for base model + specialists + large KV cache
- **4 TB SSD** — fast specialist model swapping
- **Scalable:** 2 Sparks = 256 GB (run 70B+), 4 Sparks = 512 GB (run 700B fine-tuning)
- **Zero data egress** — everything stays local
- **Unlimited queries** — no rate limits, no API quotas

### Cost Break-Even

At heavy agentic workload (~100K+ requests/day), self-hosting saves 60-80% vs API pricing. For JARVIS's use case (always-on agent backend), break-even is measured in weeks, not months.

---

## 6. Application Landscape

| Application | Why JARVIS Fits | Frontier Gap Tolerance |
|-------------|----------------|----------------------|
| HEP research agent (GRACE) | Primary mission. HEP LoRA + RAG = better than any API | High |
| Personal coding agent | SWE-bench 72%+, S* verification, runs 24/7 | Medium |
| Multi-agent orchestration | Tool use/instruction following beats frontier | Low (strength) |
| Lab notebook / experiment copilot | Local, always-on, domain-adapted | High |
| Code review / CI integration | Free per-PR analysis | Medium |
| Paper writing assistant | RAG over own papers + physics knowledge | Medium |
| Student/TA homework assistant | Physics + math + code, unlimited | High |
| Detector design exploration | GRACE + JARVIS iterating geometries 24/7 | High |
| General-purpose personal assistant | Replaces ChatGPT/Claude for daily use | Medium |

---

## 7. Open Questions

1. **GRPO vs verifier upgrade:** Which gives more ROI per SU spent?
2. **Multi-pass reasoning:** How to prototype this efficiently? Start with a simple 2-pass "attempt → reflect → re-attempt" and measure gains?
3. **Tool-calling template:** Qwen3.5's stop-word issue needs solving before any agent framework works reliably
4. **Qwen3-Coder as separate adapter?** Or is base Qwen3.5-27B good enough for coding?
5. **Multi-Spark path:** When does it make sense to buy a second Spark?
6. **Publication:** Is the JARVIS inference amplification system its own paper?

---

*This is a living document. Update as experiments complete and new ideas emerge.*
