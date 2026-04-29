# JARVIS Model Registry

All models used in JARVIS, their sources, sizes, and deployment details.

---

## Core Brain

### Qwen3.5-27B (Unified Base)

| Field | Value |
|-------|-------|
| **Model** | `Qwen/Qwen3.5-27B` |
| **Parameters** | 27B dense |
| **Architecture** | Qwen3.5 (dense + Gated Delta Networks) |
| **FP4 Size** | ~14 GB |
| **Context** | 262K native, extensible to 1M |
| **Source** | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B) |
| **License** | Apache 2.0 |
| **Training** | Optional HEP-specific LoRA (see below) |
| **Roles** | Physics, Math, Code, General |
| **Benchmarks** | GPQA Diamond: 86%, AIME 2026: 81%, LiveCodeBench: 80.7% |
| **Notes** | Single model replaces previous dual-base setup (R1-Distill-Qwen-32B + Qwen2.5-Coder-32B) |

**Why a single base:** Qwen3.5-27B exceeds our original targets across all benchmarks out-of-the-box. A single 14 GB model (FP4) replaces two 16 GB models, freeing ~18 GB of RAM for specialists, KV cache, and longer context. The inference amplification layer (best-of-N, ThinkPRM, S* verification) provides additional accuracy gains on top of the strong baseline.

### LoRA Adapters (HEP-Specific)

| Adapter | Base Model | Purpose | Size |
|---------|-----------|---------|------|
| `hep_physics` | qwen35_27b | Particle physics, detector design, scintillator properties, kinematics | 0.3 GB |
| `hep_code` | qwen35_27b | Geant4, ROOT, Pythia8, GDML patterns, HEP analysis idioms | 0.3 GB |

These adapters are hot-swapped at runtime when the router detects HEP-specific content. General physics/math/code queries use the base model without any adapter.

### Previous Models (Deprecated)

| Model | Former Role | Replaced By | Reason |
|-------|------------|-------------|--------|
| R1-Distill-Qwen-32B | Physics/Math brain | Qwen3.5-27B | Lower GPQA (61% vs 86%), larger footprint |
| Qwen2.5-Coder-32B-Instruct | Code brain | Qwen3.5-27B | Lower LiveCodeBench (55% vs 81%), larger footprint |
| R1-Distill-Llama-70B | Math brain (optional) | Qwen3.5-27B | 35 GB footprint for marginal math gain |

---

## Infrastructure Models

### Router

| Field | Value |
|-------|-------|
| **Model** | `bert-base-uncased` (fine-tuned) |
| **Parameters** | ~110M |
| **FP4 Size** | ~0.06 GB |
| **Purpose** | Two classifiers: domain (for specialist dispatch, RAG, HEP LoRA) + difficulty (easy/medium/hard for inference strategy) |
| **Training** | Fine-tuned on ~5K examples/domain + auto-generated difficulty labels |

### ThinkPRM Verifier

| Field | Value |
|-------|-------|
| **Model** | `PRIME-RL/ThinkPRM-1.5B` (or similar off-shelf PRM) |
| **Parameters** | 1.5B |
| **FP4 Size** | ~0.8 GB |
| **Purpose** | Scores reasoning chains for pessimistic verification on hard queries |
| **Source** | [HuggingFace](https://huggingface.co/PRIME-RL) |
| **Training** | None — off-the-shelf |
| **Note** | Verify compatibility with Qwen3.5's reasoning format (`<think>` tags) |

### Draft Model (Speculative Decoding)

| Field | Value |
|-------|-------|
| **Model** | `Qwen/Qwen3.5-0.8B` |
| **Parameters** | 0.8B |
| **FP4 Size** | ~0.4 GB |
| **Purpose** | Proposes draft tokens for speculative decoding (2-3x throughput) |
| **Note** | Same Qwen3.5 architecture as base model. Alternative: Qwen3.5-27B has native MTP (Multi-Token Prediction) heads, which provides speculative decoding without a separate draft model. |

### RAG Embedding Model

| Field | Value |
|-------|-------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Parameters** | 22M |
| **Size** | ~0.09 GB |
| **Purpose** | Embeds physics queries and knowledge passages for FAISS retrieval |

---

## Specialist Models

### ESM3-open (Proteins)

| Field | Value |
|-------|-------|
| **Model** | `EvolutionaryScale/esm3-sm-open-v1` |
| **Parameters** | 1.4B |
| **Size** | ~0.7 GB (FP16, small enough for full precision) |
| **Purpose** | Protein sequence/structure/function reasoning |
| **Type** | Non-text (protein language model — requires custom adapter) |
| **Source** | [HuggingFace](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) |
| **License** | EvolutionaryScale open license |
| **Key capability** | Joint reasoning over sequence, structure, and function |
| **Load policy** | On-demand from SSD |

### Evo 2 (Genomics/DNA)

| Field | Value |
|-------|-------|
| **Model** | `arcinstitute/evo2-7b` |
| **Parameters** | 7B |
| **FP4 Size** | ~3.5 GB |
| **Purpose** | DNA sequence modeling, mutation effect prediction, genome design |
| **Type** | Non-text (DNA language model — requires custom adapter) |
| **Source** | [GitHub](https://github.com/ArcInstitute/evo2) |
| **License** | Apache 2.0 |
| **Key capability** | Single-nucleotide resolution, 1M bp context window |
| **Load policy** | On-demand from SSD |
| **Note** | 40B version available (~20 GB FP4) for higher quality if memory allows |

### ChemLLM-7B (Chemistry)

| Field | Value |
|-------|-------|
| **Model** | `AI4Chem/ChemLLM-7B-Chat` |
| **Parameters** | 7B |
| **FP4 Size** | ~3.5 GB |
| **Purpose** | Chemistry QA, reaction prediction, molecular property queries |
| **Type** | Text LLM (standard chat format) |
| **Source** | [HuggingFace](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat) |
| **License** | Apache 2.0 |
| **Load policy** | On-demand from SSD |

### BioMistral-7B (Biomedicine)

| Field | Value |
|-------|-------|
| **Model** | `BioMistral/BioMistral-7B` |
| **Parameters** | 7B |
| **FP4 Size** | ~3.5 GB |
| **Purpose** | Biomedical QA, clinical reasoning, PubMed knowledge |
| **Type** | Text LLM (standard chat format) |
| **Source** | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B) |
| **License** | Apache 2.0 |
| **Load policy** | On-demand from SSD |

### Deferred Backends (Archived to External SSD)

These models are downloaded and stored on `D:\jarvis-models\` but are not runnable on the current DGX Spark target (128 GB unified RAM). They are kept locally as future-proofing — once hardware is upgraded, flipping `load_policy` in `configs/models.yaml` activates them.

| Model | Source | Storage Size | Activated / Total | Hardware Required | Status |
|-------|--------|--------------|-------------------|--------------------|--------|
| **DeepSeek-V4-Flash** | `deepseek-ai/DeepSeek-V4-Flash` | ~150 GB | 13B / 292B (MoE) | Mac Studio M3 Ultra 256GB, 2× H100 80GB, or DGX Station | Archived, plug-in ready |
| **DeepSeek-V4-Pro** | `deepseek-ai/DeepSeek-V4-Pro` | ~850 GB | 49B / 1.6T (MoE) | DGX Station GB300, 8× H100, or 4× Mac Studio cluster | Archival only |
| **DeepSeek-V4-Flash MLX 4-bit** | `mlx-community/DeepSeek-V4-Flash-4bit` | ~145 GB | 13B / 292B (MoE) | Apple Silicon (M3/M4 Ultra) | Optional |

**Key facts about V4:**

- Released 2026-04-24 (preview). MIT-licensed open weights.
- 1M token context via hybrid attention (CSA + HCA): 27% inference FLOPs and 10% KV cache vs V3.2 at 1M context.
- Trained on 32T tokens. Mixed FP4+FP8 native precision.
- **Sub-Q4 quantization is NOT viable** — experts already at 4-bit, no headroom. Native FP4 is the smallest safe form.
- Benchmarks: V4-Pro LiveCodeBench 93.5, IMOAnswerBench 89.8, AIME ~94. Beats Claude Opus 4.6 on most coding/math.

**Why archive locally:** DeepSeek has pulled weights from HF before. MIT-licensed mirrors will exist but aren't guaranteed at the official path. Local archive guarantees access regardless.

**Download:** `pwsh scripts\archive_v4_to_ssd.ps1 -Preflight` then `-Phase 1/2/3` (see script header).

---

### Future Specialists (Not Yet Integrated)

| Model | Domain | Params | FP4 Size | Source | Priority |
|-------|--------|--------|----------|--------|----------|
| OpenBioLLM-70B | Biomedicine | 70B | 35 GB | HuggingFace | Low (memory-heavy) |
| SaulLM-7B | Legal | 7B | 3.5 GB | HuggingFace | Low (not HEP-relevant) |
| FinGPT (LoRA) | Finance | LoRA | 0.3 GB | GitHub | Low |
| GeoGalactica | Geoscience | 30B | 15 GB | HuggingFace | Medium |
| BioMedLM | Biomedicine | 2.7B | 1.4 GB | HuggingFace | Medium (tiny footprint) |
| Evo 2 40B | Genomics | 40B | 20 GB | GitHub | Medium (higher quality) |

---

## Total Storage Requirements

| Category | Size | Location |
|----------|------|----------|
| Core brain (FP4) | ~14 GB | SSD (loaded to RAM) |
| Infrastructure models | ~1.8 GB | SSD (always in RAM) |
| Specialist models (FP4) | ~11-15 GB | SSD (loaded on demand) |
| LoRA adapters | ~0.6 GB | SSD (hot-swapped) |
| RAG FAISS index | ~5 GB | SSD (always in RAM) |
| **Total model storage** | **~33-37 GB** | |
| **4 TB SSD capacity** | **4,000 GB** | Room for 100+ additional models |

**Memory headroom:** With only ~21 GB always-resident (vs ~50 GB previously), the DGX Spark has ~90 GB available for specialists, KV cache, and concurrent inference. This enables longer context windows, more parallel best-of-N candidates, or loading multiple specialists simultaneously.
