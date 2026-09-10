# JARVIS — Next Steps

**Written:** 2026-09-09
**Context:** Resuming after a four-month gap (last commit 2026-05-05). This file is the
pick-up point. Read it before touching Delta or spending SUs.

**Supporting research:**
- `docs/MODEL_LANDSCAPE_2026-09.md` — full open-weights survey, May→Sept 2026, with sources
- `docs/FRAMEWORK_AUDIT_2026-09.md` — hot-swap audit with file:line evidence

---

## TL;DR

1. **Do not submit Phase 5 SFT yet.** LoRA adapters are never applied at inference —
   the training would produce artifacts nothing can load.
2. **Qwen3.8-27B supersedes Qwen3.5-27B.** Same size, same license, better on every
   published benchmark. Migrate, but measure first.
3. **806 GB of dead weight on D:\.** DeepSeek V4-Pro archive is unrunnable and outdated.

---

## 1. BLOCKER — LoRA adapters are dead code in serving

**Severity: blocks the 400 SU Phase 5 spend.**

The router selects a HEP adapter, `resolve_for_routing` calls `swap_adapter`, and
`swap_adapter` writes the adapter name into `self._active_adapters`. **Nothing ever reads
that dictionary.**

- `src/jarvis/brains/brain_manager.py:114-140` — `swap_adapter` is bookkeeping only
- `src/jarvis/brains/model_loader.py:187-194` — `LLM(...)` built without `enable_lora=True`
- `src/jarvis/brains/model_loader.py:76-84` — `generate()` passes no `lora_request`
- No `LoRARequest` anywhere in `src/`

**Why the tests didn't catch it:** `tests/test_brains.py:146-181` and
`tests/test_router.py:161-168` assert only that the dictionary got set. 164 green tests
coexist with an absent feature.

**Fix:**
- `enable_lora=True` and `max_lora_rank=32` in `load_model()`
- Thread `_active_adapters[base_key]` through `generate()` / `generate_stream()` as a
  `vllm.lora.request.LoRARequest`
- Add a test that asserts generation output differs with vs. without an adapter

---

## 2. Related pre-flight bugs in the training path

Found while auditing `scripts/run_hep_sft.sh` against `training/physics/run_sft.py`.
All must be fixed before spending the 400 SU.

| # | File | Problem |
|---|------|---------|
| 1 | `training/physics/run_sft.py:110-120` | **Padding tokens are trained on.** `labels = input_ids.clone()` masks only the prompt. `padding="max_length"` at 8192 + `pad_token = eos_token` means a 2000-token trace contributes ~6000 loss-bearing EOS labels. Model learns to emit EOS immediately. Fix: `labels[attention_mask == 0] = -100`. |
| 2 | `training/physics/run_sft.py:79` | `padding="max_length"` costs a full 8192-token forward pass per example regardless of true length. Dynamic padding + length grouping cuts wall-clock (and SU) substantially. |
| 3 | `training/eval/base.py:100-113` | `enable_lora=True` with no `max_lora_rank`. vLLM defaults to 16; training uses rank 32. Post-SFT eval fails to load the adapter after ~15h of training. |
| 4 | `scripts/run_hep_sft.sh:120` | `--use_dora true`. Confirm vLLM in the venv actually serves DoRA adapters (magnitude vectors). If unsure, plain LoRA r=32 is the safe choice. Affects serving, not just eval. |
| 5 | `training/physics/run_sft.py:190-193` | `model.save_pretrained()` called outside the Trainer under ZeRO-3, from all 4 ranks, into one directory. Recoverable via the shell script's last-checkpoint fallback, but fragile. |

---

## 3. Model migration — Qwen3.5-27B → Qwen3.8-27B

Released 2026-08-13/14, Apache 2.0, dense 27B. Official model cards, apples to apples:

| Benchmark | Qwen3.5-27B | Qwen3.8-27B |
|---|---|---|
| GPQA Diamond | 85.5 | **89.2** |
| LiveCodeBench v6 | 80.7 | **90.3** |
| AIME 2026 | 90.83 | not published |
| SWE-bench Verified | 72.4 | withdrawn (see §5) |
| SWE-bench Pro | not published | 61.7 |
| IFBench | — | 79.5 |

- 64 layers, hidden 5120, hybrid Gated DeltaNet + Gated Attention
- 262,144 native context, extensible to ~1M
- **Natively multimodal** (image + video, explicit STEM-diagram support) — relevant to
  GRACE `paper_ingest`
- Unsloth reports the config declares the `qwen3_5` architecture, so existing LoRA target
  modules (`q,k,v,o,gate,up,down`) carry over unchanged

**Our harness is trustworthy.** Measured baselines were GPQA 85.4 / AIME 89.2 /
LiveCodeBench 82.5 against official 85.5 / 90.83 / 80.7 — within ~1.5 points. So measuring
Qwen3.8 ourselves will give a reliable answer.

### Migration risks

- HF discussion threads report the model **failing to stop thinking** ("This model cannot
  stop thinking", "A crazy thinking model")
- Independent measurement: **~2× tokens per task** vs Qwen3.6. Inflates trace-gen SU cost
  and interacts badly with the 8192-token SFT cap.
- **Chat template changed.** `enable_thinking` toggle replaced by `reasoning_effort`
  (`low` / `medium` / `xhigh`), default `xhigh`.
- **Sampling changed.** Qwen3.8 thinking mode wants `temperature=1.0, top_p=0.95, top_k=20`.
  Qwen3.5 used 0.6.

### Trace regeneration

The 5000 physics + 1904 code filtered traces on Delta were generated *by Qwen3.5-27B*.
Training Qwen3.8 on them teaches a stronger model to imitate a weaker teacher. Regenerate
(~100 SU) before the 400 SU SFT if we migrate.

### Doc discrepancy to check

`CLAUDE.md` calls Qwen3.5-27B "dense", but its model card describes gated delta networks
with sparse MoE. Changes what the LoRA target modules actually attach to. Worth confirming.

---

## 4. Framework hot-swap audit

**Config layer: good.** `configs/models.yaml` drives the registry; `src/jarvis/config.py`
validates adapters against declared base models with Pydantic. Swapping the base model is
close to a config change, as intended.

**Serving layer: does not honor it.**

- **LoRA never applied** (see §1)
- **Thinking-format parsing hardcoded in 5 places**, none config-driven:
  `src/jarvis/inference/thinking.py:12-21`, `budget_forcing.py:63-66`, `voting.py:17-18`,
  `training/physics/run_sft.py:87-89`, `training/eval/base.py:171-175`
- **The `architecture:` field in `models.yaml` is declared and validated but never
  influences behavior.** Swapping model families means editing Python.
- **Those 5 places already disagree.** `thinking.py` documents Qwen3.5 as emitting visible
  `Thinking Process:` text; `training/eval/base.py:171-175` documents the same model as
  emitting `</think>` tags and strips on them. Per memory, the first is correct — so the
  eval path strips on a token that never appears.
- **Sampling deviates from published protocol.** `top_k` is absent entirely from
  `model_loader.py:_build_sampling_params`, while both Qwen cards specify 20. The eval
  harness sets it correctly, so serving and eval run different protocols.

**Recommended refactor (no compute):** move thinking-format regexes, stop strings, and
sampling defaults into a per-architecture block in `configs/models.yaml`, keyed by the
existing `architecture:` field. Then a model swap really is a config change.

---

## 5. Benchmark hygiene

- **SWE-bench Verified was withdrawn in Feb 2026 over contamination.** Our
  `training/eval/run_swebench.py` + `scripts/run_swebench_mini.sh` target a deprecated
  benchmark. Migrate to **SWE-bench Pro**.
- **CritPt** is now the research-level physics benchmark — 71 challenges from 50+
  physicists across 30 institutions, explicitly covering high energy physics. Far better
  positioning for the GRACE paper than GPQA alone. Eval API access is granted case-by-case
  to researchers, free.
- **SciCode** (338 subproblems, 16 scientific subfields) is the scientific-code companion.
- Both are in Artificial Analysis's current index suite.

---

## 6. D:\ drive cleanup (2 TB SSD, 808 GB free)

| Path | Size | Verdict |
|------|------|---------|
| `jarvis-models/deepseek/v4-pro-fp4` | **806 GB** | **Delete.** Unrunnable on any owned or planned hardware. Superseded by V4-Pro-0813 (2026-08-13). |
| `jarvis-models/deepseek/v4-flash-fp4` | 149 GB | Superseded by **V4-Flash-0731** (re-post-trained, +10 AA index, MIT, same 284B/13B-active architecture). Refresh or drop. |
| `jarvis-models/qwen/qwen3.5-27b-fp4` | 52 GB | Current base. Keep until Qwen3.8 migration lands. |
| `jarvis-models/specialists` | 41 GB | Keep (ChemLLM, BioMistral, Evo 2). |
| `jarvis-models/infrastructure` | 6 GB | Keep (ThinkPRM, draft model, RAG embedder). |
| `jarvis-models/adapters` | **empty** | Confirms Phase 5 never produced adapters. |
| `D:\src` | 952 MB | Older duplicate of `D:\jarvis-agent-body` (1,905 files differ). Delete. |
| `D:\jarvis-body.zip`, `D:\codeagent_src.zip` | ~21 MB | Redundant. Delete. |
| `D:\jarvis-agent-body` | 969 MB | **Keep** — this is the good copy, has `guide/`. |

**Nothing has been deleted.** All of the above needs explicit approval.

Also fix:
- `docs/CODEAGENT_INTEGRATION.md` and memory refer to `D:\jarvis-body` — the real path is
  `D:\jarvis-agent-body`.
- `D:\jarvis-models\MANIFEST.md` claims 1052.5 GB total and lists a `v4-flash-mlx-4bit`
  copy that does not exist. Actual total is ~1.05 TB across two DeepSeek dirs only.
- `configs/models.yaml` `deferred_backends:` still points at the superseded V4 preview
  repos (`deepseek-ai/DeepSeek-V4-Flash`, `-V4-Pro`). Update to the `-0731` / `-0813` repo
  IDs if we keep them.

---

## Recommended order of work

| # | Task | Cost | Blocks |
|---|------|------|--------|
| 1 | Fix LoRA serving path (§1) | none | Phase 5 |
| 2 | Fix SFT training bugs 1–5 (§2) | none | Phase 5 |
| 3 | Delete `v4-pro-fp4`, free 806 GB (§6) | none | step 4 |
| 4 | Download Qwen3.8-27B, run GPQA + AIME on Delta | ~25 SU | migration decision |
| 5 | If §4 confirms the gain: regenerate traces on Qwen3.8 | ~100 SU | step 6 |
| 6 | Phase 5 SFT — `run_hep_sft.sh --physics`, then `--code` | ~400 SU | Phase 6 |
| 7 | Config-drive thinking format + sampling (§4) | none | future swaps |
| 8 | Migrate SWE-bench harness to Pro; look at CritPt (§5) | small | paper claims |

**Budget:** ~7,900 SU remaining. Steps 4–6 total ~525 SU.

---

## Open items carried over from Phase 4 (still unresolved)

From `project_hep_lora_pipeline.md`:

1. `extract_hep_physics.py::extract_constraint_problems` — regex doesn't match real schema, returns 0
2. `extract_hep_physics.py::extract_paper_problems` — JSON shape mismatch with `benchmark_papers.json`
3. `extract_hep_code.py::extract_grace_tools_yaml_problems` — YAML path likely wrong, returns 0
4. **Bug A (MMLU answers):** `download_benchmarks.py::download_mmlu_stem` drops the MMLU
   `answer` field, so all 4920 MMLU traces land in "no GT, kept blindly". ~1h to fix, no SU.
5. `source` field is `?` in all raw traces — `generate_traces_api.py` doesn't propagate
   `problem.source`. Blocks stratified sampling. Cosmetic.

**Harmless noise:** every Python invocation on Delta prints a
`matplotlib-3.8.0.dev452+g66ba515e6-nspkg.pth` AttributeError. Stale editable install in
the venv. Zero functional effect.

---

## Delta state as of 2026-05-05 (VERIFY BEFORE USE)

`/work/hdd` is scratch and carries a purge policy. Four months have passed. **Confirm these
still exist before planning around them:**

```bash
ssh jhill5@login.delta.ncsa.illinois.edu \
  'ls -la /work/hdd/bgde/jhill5/data/hep_*_filtered.jsonl'
```

- `/work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl` — 5000 traces, 823 problems, 100 MB
- `/work/hdd/bgde/jhill5/data/hep_code_filtered.jsonl` — 1904 traces, 255 problems, 58 MB

If purged, regenerating costs ~100 SU — which is the same as step 5 above, so migrating to
Qwen3.8 would then be strictly better than restoring the old traces.

---

## Sources

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [vLLM recipe, Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Unsloth Qwen3.8 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.8/train)
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [CritPt benchmark](https://github.com/CritPt-Benchmark/CritPt)
- [SciCode benchmark](https://github.com/scicode-bench/SciCode)
- [Artificial Analysis, open-source models](https://artificialanalysis.ai/models/open-source)
