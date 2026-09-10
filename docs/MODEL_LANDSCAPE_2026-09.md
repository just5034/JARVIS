# Open-Weights Model Landscape — September 2026

**Survey date:** 2026-09-09
**Purpose:** Establish whether a better base model exists before committing ~400 SU to HEP
LoRA training on Qwen3.5-27B.
**Window covered:** 2026-05 (our last session) through 2026-09.

> **Sourcing rule used here:** benchmark numbers are taken from official model cards or the
> benchmark's own leaderboard. Third-party aggregator blogs were found to be unreliable —
> see [Corrections](#corrections) at the bottom.

---

## 1. Verdict

**Qwen3.8-27B is the only model that changes our plans.** It occupies exactly the slot
Qwen3.5-27B occupies today — same parameter count, same Apache 2.0 license, same hardware
envelope — and beats it on every benchmark both cards report.

Everything genuinely stronger this cycle went enormous and needs a cluster. Nothing new in
the 20B–40B class competes.

---

## 2. Our size class (20B–40B, runnable on DGX Spark / trainable on 4× A100-40GB)

### Qwen3.8-27B — **the pick**

| Field | Value |
|-------|-------|
| Released | 2026-08-13/14 |
| License | Apache 2.0 |
| Parameters | 27B, 64 layers, hidden 5120 |
| Architecture | Hybrid: Gated DeltaNet (linear attention) + Gated Attention |
| Context | 262,144 native, extensible to ~1M |
| Multimodal | Yes — image + video, explicit STEM-diagram and document support |
| Weights | ~56 GB BF16, ~31 GB FP8, ~14–16 GB 4-bit (before KV cache) |
| Thinking | On by default; `reasoning_effort` = `low` / `medium` / `xhigh` (default `xhigh`) |
| Sampling (thinking) | `temperature=1.0, top_p=0.95, top_k=20` |
| Sampling (instruct) | `temperature=0.7, top_p=0.80, presence_penalty=1.5` |

**Head-to-head against our current base**, official cards only:

| Benchmark | Qwen3.5-27B | Qwen3.8-27B | Δ |
|---|---|---|---|
| GPQA Diamond | 85.5 | **89.2** | +3.7 |
| LiveCodeBench v6 | 80.7 | **90.3** | +9.6 |
| AIME 2026 | 90.83 | not published | — |
| SWE-bench Verified | 72.4 | withdrawn (§5) | — |
| SWE-bench Pro | not published | 61.7 | — |
| IFBench | not published | 79.5 | — |
| MMLU-Pro | 86.1 | not published | — |

The two cards report different coding benchmarks because SWE-bench Verified was withdrawn
between the releases. The intermediate Qwen3.6-27B was reported at 77.2 on Verified (vs
3.5's 72.4) and 53.5 on Pro (vs 3.8's 61.7), so the coding trend is upward across all three.

**Migration is mechanically cheap.** Unsloth reports the Qwen3.8 config declares the
`qwen3_5` architecture string, and recommends exactly the target modules we already use:
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. No change to
`scripts/run_hep_sft.sh`'s `--lora_target_modules`.

**Known risks:**

- Hugging Face discussion threads titled *"This model cannot stop thinking"* and *"A crazy
  thinking model"* report runaway reasoning.
- Independent measurement puts token usage at **~2× per task** vs Qwen3.6. This inflates
  trace-generation SU cost and pushes against our 8192-token SFT sequence cap.
- Chat template changed — the `enable_thinking` toggle is replaced by `reasoning_effort`.
- Recommended temperature moved from 0.6 to 1.0.
- Gated DeltaNet kernels compile on first run; expect a slow first job.

**Why it matters beyond the numbers:** native vision-language with explicit STEM-diagram
support is directly relevant to GRACE's `paper_ingest` tool (plots, detector schematics,
figures in PDFs). Qwen3.5-27B is also multimodal, so this is an improvement in degree, not
a new capability.

### Everything else in this class — rejected

| Model | Released | Why not |
|---|---|---|
| gpt-oss-120b / -20b | 2025-08 | A year old. 120b reports GPQA-D 80.1, below both Qwen 27Bs. MoE, 5.1B active. |
| Gemma 4 (5 sizes, incl. 26B-A4B) | 2026-03-31 | Apache 2.0 now, but the 26B variant activates only 3.8B params. Not competitive at our quality bar. |
| Mistral Small 4 | 2026-03-16 | 119B total / 6B active MoE, 256K context. Benchmarks below the Qwen dense line. Predates our last session; we already passed on it. |
| Phi-4 14B | earlier | Strong math-per-parameter but well below our current baseline overall. |

---

## 3. Larger open weights — do not fit, tracked for awareness

None of these run on 128 GB unified RAM or train on 4× A100-40GB.

| Model | Params | Released | License | Note |
|---|---|---|---|---|
| Kimi K3 | 2.8T total / 104B active | weights 2026-07-27 | Free commercial **with conditions** | 1.56 TB. Revenue-tiered: MaaS businesses over $20M/12mo must sign a separate agreement; products over 100M MAU must display "Kimi K3" in the UI. Leads open weights on GPQA and AIME among open models. |
| GLM-5.3 | 743–753B / 40B active | 2026-08-14, weights 08-25 | Custom GLM-5.3 license | Same base as GLM-5.2, upgraded via post-training only. |
| GLM-5.2 | 753B / 40B active | 2026-06-13 | **MIT** | SWE-bench Pro 62.1. The permissive one of the pair. |
| GLM-5.3-Flash | 320B / 18B active | 2026-08 | Custom | Smaller sibling; still far past our envelope. |
| Nemotron 3 Ultra | 550B / 55B active | 2026-06-04 | Open weights | NVIDIA. Hybrid Mamba-attention MoE, 1M context. Publishes training data, recipes, reward models, NVFP4 variants. |
| DeepSeek V4-Pro-0813 | ~1.6T / 49B active | 2026-08-13 | **MIT** | SWE-bench Verified 80.6, strongest open-weight on that (now-deprecated) benchmark. |
| DeepSeek V4-Flash-0731 | 284B / 13B active | 2026-07-31 | **MIT** | See §4. |
| Qwen3.8-Flash-Next | 125B + 51B n-gram embedding | 2026-08-26 | Apache 2.0 | **Does not fit 128 GB** despite 6B active params — the full ~176B must be resident. FP8 checkpoint is 172.78 GiB; vLLM recipe budgets 250 GB. Marketed as a Qwen4 architecture preview. |
| Qwen3.8-Max / 2.4T A95B | 2.4T / 95B active | 2026-08-03 | — | Flagship. |

### Qwen3-Coder-Next — worth a second look

| Field | Value |
|---|---|
| Released | 2026-02-04 |
| License | Apache 2.0 |
| Parameters | 80B total / **3B active** MoE |
| Context | 256K |
| SWE-bench Verified | 74.2 |
| SWE-bench Pro | 44.3 |
| SWE-bench Multilingual | 63.7 |

At ~40 GB in 4-bit this **fits DGX Spark**, and 3B active parameters makes it fast. It is
already named in `project_coding_tier1_plan.md` as a specialist candidate. Its SWE-bench
Pro score (44.3) is below Qwen3.8-27B's (61.7), so it is not a base-model candidate — but
it remains interesting as a cheap, high-throughput code specialist.

---

## 4. DeepSeek V4 — our archive is a superseded checkpoint

We archived V4-Flash and V4-Pro to `D:\jarvis-models\` on 2026-04-28. **DeepSeek has since
republished both.**

| | Archived (preview) | Current |
|---|---|---|
| Flash | `deepseek-ai/DeepSeek-V4-Flash` | `deepseek-ai/DeepSeek-V4-Flash-0731` (2026-07-31) |
| Pro | `deepseek-ai/DeepSeek-V4-Pro` | `DeepSeek-V4-Pro-0813` (2026-08-13) |

**V4-Flash-0731** keeps the same architecture and size — 284B total / 13B active, 1M
context, 384K max output, MIT — and was **re-post-trained** for agentic work. It ships with
DSpark, a speculative-decoding module that can be enabled at serving time, and exposes
`reasoning_effort` at `low` / `high` / `max`.

Reported gains over the preview: Terminal-Bench 2.1 up 17 points to 79%, τ³-Bench Banking
up 8 points to 31%, GDPval-AA v2 Elo 1189 → 1559. It reportedly beats V4-Pro (preview)
despite far fewer activated parameters.

**Implications for us:**
- The 149 GB Flash copy on D:\ is the old checkpoint. Refresh it or drop it.
- `configs/models.yaml` `deferred_backends:` points at the superseded repo IDs.
- Neither fits 128 GB. This remains archive hygiene, not a deployment change.

---

## 5. Benchmark hygiene — what changed under us

**SWE-bench Verified was deprecated in February 2026 over contamination.** OpenAI withdrew
it and the community has moved to **SWE-bench Pro**. Our `training/eval/run_swebench.py`
and `scripts/run_swebench_mini.sh` target the deprecated benchmark. Our recorded 50% on a
24-instance astropy-heavy sample is therefore measured against a benchmark nobody quotes
any more.

Two benchmarks now carry weight for the GRACE positioning and we track neither:

**CritPt** (Complex Research using Integrated Thinking – Physics Test)
- 71 challenges (70 test + 1 example), built by 50+ active physics researchers across 30
  institutions, ~40 review-hours per challenge
- Covers 11 subfields **including high energy physics**, plus condensed matter, quantum,
  AMO, astrophysics, statistical, nuclear, mathematical physics, fluids, nonlinear
  dynamics, biophysics
- Guess-resistant answer formats: floating-point arrays, symbolic expressions, Python
  functions
- Frontier models score around 32%; 2025-era models were in single digits
- Eval pipeline and dataset are public; Artificial Analysis runs a grading API, free, with
  case-by-case access for researchers

**SciCode**
- 338 subproblems decomposed from 80 main problems across 16 subfields of physics, math,
  chemistry, biology, materials
- Scientist-annotated gold solutions and test cases
- Integrated with `inspect_ai`; 273 models evaluated
- Top scores are in the high 50s to low 60s

Both are in Artificial Analysis's current index suite alongside Humanity's Last Exam,
Terminal-Bench v4.0, GDPval-AA v2 and AA-LCR. For a paper claiming a physics-specialized
agent, CritPt is a far stronger yardstick than GPQA Diamond, which is graduate-exam-shaped
rather than research-shaped.

---

## 6. Tool calling / agentic

- **BFCL v4** (April 2026) shifted to holistic agentic evaluation: Agentic 40%, Multi-Turn
  30%, Live 10%, Non-Live 10%, Hallucination 10%.
- The gap between frontier closed APIs and top open-weight models on BFCL v4 has narrowed
  to roughly 3–4 points.
- Convention is now to quote **BFCL plus τ-Bench** together — BFCL for the function-calling
  primitive, τ-Bench for production-shaped multi-turn integration.

Relevant to `docs/CODEAGENT_INTEGRATION.md`: our tool-call path depends on vLLM's
`qwen3_xml` parser and Qwen3.5's XML `<tool_call>` emission. **Re-verify that parser choice
against Qwen3.8 before assuming the shim still works.**

---

## 7. Specialists — no action needed

Checked for successors to our registered specialists. Nothing forces a change.

- **ESM3** — still current for controllable protein generation. **ESM Cambrian** is a
  parallel family focused on representation rather than generation, not a successor. ESM3
  open weights remain gated on HF; we have not requested access.
- **Evo 2** — 40B, 1M-token single-nucleotide context. Still the reference genomics model.
- ChemLLM-7B and BioMistral-7B unchanged.

---

## 8. Our harness is trustworthy

Worth recording, because it determines how much we can rely on our own measurements.

| Benchmark | Our measured (Qwen3.5-27B, Delta) | Official card | Δ |
|---|---|---|---|
| GPQA Diamond | 85.4 | 85.5 | −0.1 |
| AIME 2026 | 89.2 | 90.83 | −1.6 |
| LiveCodeBench v6 | 82.5 | 80.7 | +1.8 |

Within ~1.5 points across the board. **We can measure Qwen3.8-27B ourselves and trust the
result** rather than relying on published numbers.

The SWE-bench outlier (50% measured vs 72.4 published) is a scaffolding gap, not a model
gap — and against a deprecated benchmark besides.

---

## 9. Corrections

Recorded so the error is not repeated.

**Artificial Analysis Intelligence Index.** An earlier reading of this survey cited
Qwen3.8-27B at 52 and Qwen3.6-27B at 38 on the AA Intelligence Index, sourced from a
third-party blog. **That is wrong.** Artificial Analysis does not score the 27B dense model
in its open-weights ranking at all, and the index was recalibrated onto a harder eval suite
in 2026. AA's own published open-weights ranking reads GLM-5.3 at 45, Kimi K3 at 44,
GLM-5.3-Flash at 42, Qwen3.8 2.4T A95B at 40, DeepSeek V4-Pro-0813 at 36. Frontier closed
models sit around 53.

**Lesson:** for this model generation, third-party aggregator sites disagree with each
other and with primary sources. Use official model cards and the benchmarks' own
leaderboards.

**Doc discrepancy to resolve.** `CLAUDE.md` and `docs/MODELS.md` describe Qwen3.5-27B as
"dense". Its model card describes gated delta networks combined with sparse
mixture-of-experts. This changes what LoRA target modules actually attach to and should be
confirmed against the config on disk.

---

## Sources

**Primary (model cards / benchmark repos):**
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
- [CritPt-Benchmark/CritPt](https://github.com/CritPt-Benchmark/CritPt)
- [scicode-bench/SciCode](https://github.com/scicode-bench/SciCode)
- [MathArena](https://matharena.ai/)

**Deployment / tooling:**
- [vLLM recipe — Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM recipe — Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
- [Unsloth — Qwen3.8 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.8/train)
- [vLLM DoRA support PR #14389](https://github.com/vllm-project/vllm/pull/14389)
- [vLLM DoRA feature request #10849](https://github.com/vllm-project/vllm/issues/10849)

**Leaderboards / analysis:**
- [Artificial Analysis — open source models](https://artificialanalysis.ai/models/open-source)
- [Artificial Analysis — CritPt](https://artificialanalysis.ai/evaluations/critpt)
- [Artificial Analysis — SciCode](https://artificialanalysis.ai/evaluations/scicode)
- [NVIDIA Nemotron 3 Ultra technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
- [Kimi K3 specs](https://www.datalearner.com/en/ai-models/pretrained-models/kimi-k3)
- [GLM-5.2 overview](https://www.morphllm.com/glm-5-2)
- [Mistral Small 4 announcement](https://mistral.ai/news/mistral-small-4/)
