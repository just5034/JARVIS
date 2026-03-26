# JARVIS Model Registry

All models used in JARVIS, their sources, sizes, and deployment details.

---

## Core Brains

### Math Brain

| Field | Value |
|-------|-------|
| **Primary** | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` |
| **Fallback** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (math LoRA on physics base) |
| **Parameters** | 70B (primary) / 32B (fallback) |
| **FP4 Size** | ~35 GB (primary) / ~16 GB shared + 0.3 GB LoRA (fallback) |
| **Source** | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| **License** | DeepSeek License (permissive, allows commercial) |
| **Training** | None — off-the-shelf |
| **Benchmarks** | MATH-500: 94.5%, AIME 2024: 70.0%, GPQA: 65.2% |
| **Notes** | With consensus@16 + Heimdall verification, AIME reaches ~87-90% |

### Physics Brain

| Field | Value |
|-------|-------|
| **Base** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| **Adapter** | Custom LoRA trained on Delta (physics_general + physics_hep) |
| **Parameters** | 32B base + LoRA adapters |
| **FP4 Size** | ~16 GB (Qwen2.5 base, always resident) + 0.3 GB (adapter) |
| **Source** | Base: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| **License** | MIT (base model) |
| **Training** | ~4,200 SU on Delta — distillation SFT + GRPO RL + ETTRL |
| **Baseline** | GPQA Diamond: 62.1% |
| **Target** | GPQA Diamond: 78-84% |

### Code Brain

| Field | Value |
|-------|-------|
| **Base** | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| **Adapter** | Custom LoRA trained on Delta (code_hep only — base model already strong at general code) |
| **Parameters** | 32B base + LoRA adapter |
| **FP4 Size** | ~16 GB (Qwen2.5-Coder base, always resident) + 0.3 GB (adapter) |
| **Source** | Base: [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |
| **License** | Apache 2.0 |
| **Training** | ~250 SU on Delta — HEP-specific code LoRA only (Geant4, ROOT, Pythia patterns) |
| **Baseline** | HumanEval: 88.4%, LiveCodeBench: ~40-50% |
| **Target** | LiveCodeBench: 65%+ (via S* execution verification + HEP LoRA) |

**Two separate bases, same architecture.** Physics uses R1-Distill-Qwen-32B and Code uses Qwen2.5-Coder-32B-Instruct. Both are Qwen2.5 architecture but LoRA adapters are still base-specific (trained on different model weights). Both bases are always resident in memory (~32 GB total at FP4). The math brain uses a LoRA adapter on the physics base, or optionally the separate R1-Distill-Llama-70B for maximum math performance.

**Code performance strategy:** Instead of expensive AZR self-play training (original: 2,600 SU), we use a purpose-built code model + S* execution-based verification at inference time. This achieves the same target with 250 SU and redirects 2,350 SU to physics GRPO training.

---

## Infrastructure Models

### Router

| Field | Value |
|-------|-------|
| **Model** | `bert-base-uncased` (fine-tuned) |
| **Parameters** | ~110M |
| **FP4 Size** | ~0.06 GB |
| **Purpose** | Two classifiers: domain (math/physics/code/specialist) + difficulty (easy/medium/hard) |
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

### Draft Model (Speculative Decoding)

| Field | Value |
|-------|-------|
| **Model** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| **Parameters** | 1.5B |
| **FP4 Size** | ~0.8 GB |
| **Purpose** | Proposes draft tokens for speculative decoding (2-3× throughput) |
| **Source** | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

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

## Data Teacher (Training Only — Not Deployed)

### DeepSeek R1-0528

| Field | Value |
|-------|-------|
| **Model** | `deepseek-ai/DeepSeek-R1-0528` |
| **Parameters** | 685B MoE (~37B active) |
| **Purpose** | Generate training traces for physics brain distillation |
| **Deployed locally?** | **NO** — 342 GB at FP4, exceeds DGX Spark capacity |
| **Access method** | API during training on Delta, or quantized inference on H200 cluster |
| **Benchmarks** | GPQA: 81.0%, AIME 2025: 87.5%, MATH-500: 97.3% |
| **License** | MIT |

---

## Total Storage Requirements

| Category | Size | Location |
|----------|------|----------|
| Core brains (FP4) | ~35-51 GB | SSD (loaded to RAM) |
| Infrastructure models | ~1.8 GB | SSD (always in RAM) |
| Specialist models (FP4) | ~11-15 GB | SSD (loaded on demand) |
| LoRA adapters | ~2 GB | SSD (hot-swapped) |
| RAG FAISS index | ~5 GB | SSD (always in RAM) |
| **Total model storage** | **~55-75 GB** | |
| **4 TB SSD capacity** | **4,000 GB** | Room for 50+ additional models |
