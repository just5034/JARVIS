# Framework Audit — Can JARVIS Actually Hot-Swap Models?

**Audit date:** 2026-09-09
**Question asked:** Development Principle #1 says "swapping the base model is a config
change + LoRA retrain," and #2 says "all in YAML configs, not hardcoded." Is that true?
**Answer:** The config layer delivers on it. The serving layer does not.

Line numbers refer to the tree at commit `c3efb1b`.

---

## Summary

| Area | Verdict |
|---|---|
| Config schema and validation | **Good.** Pydantic-validated, YAML-driven, adapters checked against declared bases. |
| Base model swap | **Mostly works.** Path/quantization/context all config-driven. |
| LoRA adapter application | **Broken.** Adapters are never applied at inference. |
| Thinking-format handling | **Hardcoded** in 5 files, and they disagree with each other. |
| Sampling protocol | **Hardcoded and incomplete.** `top_k` absent from serving. |
| `architecture:` field | **Declared, validated, never used.** |

---

## 1. What works

`configs/models.yaml` is the registry, and `src/jarvis/config.py` gives it a real schema.

- `BaseModelConfig` (`config.py:13-24`) — `model_id`, `architecture`, `path`, `size_gb`,
  `quantization`, `context_length`, `recommended_max_context`, `load_policy`, `roles`
- `validate_adapter_base_models` (`config.py:59-64`) — every adapter's `base_model` must
  name a declared base
- `config.py:219-224` — router `domain_to_brain` mappings validated against both the base
  model registry and the adapter registry
- `always_resident` memory accounting (`config.py:70-74`)

`model_loader.load_model` (`model_loader.py:160-205`) resolves a relative `path` against the
model dir, **falls back to the HuggingFace `model_id` if the local path is missing**, and
maps `nvfp4` / `none` / `""` to `quantization=None` for vLLM. That fallback is a genuinely
good touch — a swap works before the weights are staged locally.

`configs/router.yaml` `domain_to_brain` maps every domain to a base model key by name.
Changing the base model for all domains is a find-and-replace in one YAML block.

**So: pointing JARVIS at Qwen3.8-27B is close to a config change, as advertised.**

---

## 2. BLOCKER — LoRA adapters are never applied

This is the finding that blocks the ~400 SU Phase 5 spend.

### The chain

1. `router.py:125` picks the adapter:
   ```python
   adapter = mapping.hep_adapter if is_hep and mapping.hep_adapter else mapping.adapter
   ```
2. `brain_manager.py:190` acts on it:
   ```python
   self.swap_adapter(decision.base_model, decision.adapter)
   ```
3. `brain_manager.py:114-140` — `swap_adapter` validates the adapter exists, validates it
   was trained on this base, logs, and then:
   ```python
   self._active_adapters[base_key] = adapter_key
   ```
4. **Nothing reads `_active_adapters`.**

### Evidence

```
$ grep -rn '_active_adapters' src/
brain_manager.py:25   self._active_adapters: dict[str, str | None] = {}
brain_manager.py:48   return self._active_adapters.get(base_key)
brain_manager.py:92   self._active_adapters[model_key] = None
brain_manager.py:107  self._active_adapters.pop(model_key, None)
brain_manager.py:119  current = self._active_adapters.get(base_key)
brain_manager.py:140  self._active_adapters[base_key] = adapter_key
```

Every hit is a write, a pop, or the getter feeding line 119's own early-return. No consumer.

```
$ grep -rn 'enable_lora\|max_lora_rank\|LoRARequest' src/
(no matches)
```

- `model_loader.py:187-194` — `LLM(...)` is constructed with `model`, `max_model_len`,
  `gpu_memory_utilization`, `enforce_eager`, `quantization`, `trust_remote_code`. **No
  `enable_lora`.** vLLM cannot serve a LoRA on an engine that was not built for it.
- `model_loader.py:121` — `self._llm.generate([prompt], sampling_params)`. **No
  `lora_request`.**

### Why 164 tests pass

The tests assert the bookkeeping, not the behavior.

- `tests/test_brains.py:146-181` — four tests: same-base accepted, clear, unknown base
  raises, unknown adapter raises
- `tests/test_router.py:161-168` — two tests: HEP adapter selected for physics, for code

All six check that the right *name* reaches the dictionary. None check that generation
output changes. This is a textbook green-tests-absent-feature situation.

### Consequence

`scripts/run_hep_sft.sh --physics` and `--code` would spend ~400 SU producing
`/projects/bgde/jhill5/adapters/hep_physics` and `hep_code`. The serving stack cannot load
either. `D:\jarvis-models\adapters\` is empty, which is consistent — nothing has ever been
produced or consumed.

### Fix

1. `model_loader.load_model` — add `enable_lora=True` and `max_lora_rank=32` to the `LLM(...)`
   kwargs. Gate on whether any adapter in config names this base, so non-adapter deployments
   don't pay the overhead.
2. `LoadedModelHandle` — accept an active adapter name, resolve it to a path via
   `config.models.lora_adapters[key].path`, and pass
   `vllm.lora.request.LoRARequest(name, id, path)` into `generate()` and `generate_stream()`.
3. `brain_manager` — hand `_active_adapters[base_key]` to the handle rather than storing it
   in isolation.
4. **Add a test that asserts output differs with and without an adapter.** The current tests
   would not have caught this and will not catch a regression.

---

## 3. Related: same bug in the eval path

`training/eval/base.py:82-113` — `load_model` sets `enable_lora=True` when an adapter is
passed (correct), but never sets `max_lora_rank`. vLLM's default is 16.
`scripts/run_hep_sft.sh:118` trains at `--lora_rank 32`.

The post-SFT GPQA eval at the tail of `run_hep_sft.sh` would therefore fail to load the
adapter — **after** ~15 hours of physics training. The adapter itself survives; only the
eval dies. Still worth fixing before the run.

Separately, `run_hep_sft.sh:117` passes `--use_dora true`. vLLM's DoRA support has an open
feature request and a PR; whether it merged into the version in
`/work/hdd/bgde/jhill5/jarvis-venv` is unconfirmed. DoRA adapters carry
`lora_magnitude_vector` weights that plain-LoRA loaders reject. **Verify before training, or
fall back to plain LoRA at rank 32** — this affects serving, not just eval.

---

## 4. Thinking-format handling is hardcoded and self-contradictory

Five files parse reasoning output. None consult config.

| File | Lines | Pattern |
|---|---|---|
| `src/jarvis/inference/thinking.py` | 12-21, 66-68 | `^Thinking Process:\s*\n(.*?)(?:\n\n|\Z)` and `<think>(.*?)</think>` |
| `src/jarvis/inference/budget_forcing.py` | 63-66 | strips `</think>\s*$`, then `(\nThinking Process:.*?)\n\n.+$` |
| `src/jarvis/inference/voting.py` | 17-18 | strips both forms before answer extraction |
| `training/physics/run_sft.py` | 87-89 | **writes** `f"Thinking Process:\n{reasoning}\n\n{trace}"` |
| `training/eval/base.py` | 171-175 | `if "</think>" in text: rpartition("</think>")` |

### They disagree

- `thinking.py:3` documents: *"Qwen3.5-27B outputs reasoning as visible 'Thinking Process:'
  text blocks."*
- `training/eval/base.py:171` documents: *"Qwen3.5 outputs: `<think>...reasoning...</think>`
  final answer"*

Same model, same repo, opposite claims. Per `project_qwen35_thinking_format.md`, the first
is correct. So `training/eval/base.py`'s `extract_answer` strips on a delimiter that never
appears, and passes the full text — thinking included — to downstream extraction. Our GPQA
number still matched the official card, which suggests tail-based extraction rescued it, but
this is luck, not design.

### Qwen3.8 makes this concrete

Qwen3.8 replaces the `enable_thinking` toggle with `reasoning_effort` (`low` / `medium` /
`xhigh`, defaulting to `xhigh`). Any format assumption baked into these five files is a
migration cost paid five times.

### Fix

Move format handling into a per-architecture block in `configs/models.yaml`, keyed by the
**existing but unused** `architecture:` field:

```yaml
architectures:
  qwen3.5:
    thinking_style: "prose"          # "Thinking Process:" prefix
    thinking_delimiter: "Thinking Process:"
    sampling: {temperature: 0.6, top_p: 0.95, top_k: 20}
  qwen3.8:
    thinking_style: "prose"
    reasoning_effort: "medium"       # xhigh default causes runaway thinking
    sampling: {temperature: 1.0, top_p: 0.95, top_k: 20}
  deepseek-v4:
    thinking_style: "xml"            # <think>...</think>
    sampling: {temperature: 0.6, top_p: 0.95}
```

Then one resolver reads it and the five call sites consume the resolved policy. **This is
what makes the swap claim true.** No compute required.

---

## 5. Sampling deviates from published protocol

`model_loader.py:76-84` builds `SamplingParams` with `temperature`, `top_p`, `max_tokens`,
`stop`, `n`. **`top_k` is absent entirely**, so vLLM defaults to `-1` (disabled).

Both Qwen 27B cards specify `top_k=20`. `training/eval/base.py` sets it correctly.

**Serving and evaluation therefore run different sampling protocols against the same model.**
Benchmark numbers measured through the eval harness do not describe what the API serves.

Defaults are also wrong for the model: `GenerationRequest` (`model_loader.py:40-41`) defaults
`temperature=1.0, top_p=1.0`; `engine.py:104-105` mirrors it. Qwen3.5 thinking mode wants
`0.6 / 0.95 / 20`; Qwen3.8 wants `1.0 / 0.95 / 20`.

Hardcoded per-strategy temperatures also sit in `engine.py:197` (`0.7` for voting diversity)
and `engine.py:226` (`0.8` for code). Those are deliberate, but they belong in
`configs/inference.yaml` alongside the `domain_overrides` block that already exists — which
notably *does* define a `temperature` for `code.medium` that the code path overrides anyway.

---

## 6. Specialist path is stubbed

`brain_manager.py:141-170` — `resolve_for_routing_async` loads the specialist, logs success,
then discards it:

```python
# Wrap specialist in a LoadedModelHandle-compatible interface
# For now, fall back to default model since specialists need
# their own generate() path — the adapter handles I/O translation
# TODO: Build SpecialistModelHandle that wraps adapter + model
```

Chemistry, biology, protein and genomics queries all reach the base model. The specialist is
loaded into memory and then unused. Honestly commented, so this is known — but it means the
4 of 8 router domains that dispatch to specialists currently do nothing but waste RAM.

Not on the critical path for HEP LoRA work. Recorded so it is not mistaken for working.

---

## 7. Tool-use path

`src/jarvis/tooluse/` has hardcoded model-name defaults at `anthropic_shim.py:67`,
`anthropic_translate.py:88` and `schemas.py:67` (all `"qwen3.5-27b"`). These are cosmetic
defaults for the response envelope, not routing decisions, but they will report a stale
model name after a migration.

More substantive: the working tool-call configuration depends on vLLM's `qwen3_xml` parser,
because Qwen3.5 emits `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
rather than Hermes JSON. **Re-verify the parser choice against Qwen3.8** before assuming
`scripts/run_vllm_tooluse.sh` still works.

---

## 8. Recommended order

| # | Change | Compute | Unblocks |
|---|---|---|---|
| 1 | Apply LoRA at inference (§2) | none | Phase 5 SFT |
| 2 | `max_lora_rank=32` in eval; settle DoRA (§3) | none | post-SFT eval |
| 3 | Add `top_k` to serving `SamplingParams` (§5) | none | protocol parity |
| 4 | Config-drive thinking format + sampling per `architecture:` (§4) | none | any future swap |
| 5 | `SpecialistModelHandle` (§6) | none | 4 router domains |

Steps 1–3 are small and gate real spend. Step 4 is the one that makes Development
Principle #1 actually true.

---

## Cross-references

- Plan and priorities: `docs/NEXT_STEPS.md`
- Model landscape and the case for migrating: `docs/MODEL_LANDSCAPE_2026-09.md`
- Stated principles: `CLAUDE.md` → Development Principles
- Component contracts: `docs/ARCHITECTURE.md`
