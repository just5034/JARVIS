# JARVIS Training Pipeline — NCSA Delta

**Platform:** NCSA Delta (ACCESS-CI allocation)
**Budget:** 8,000 Service Units (SUs) — ~76 SU spent, ~7,924 remaining
**Login:** `ssh login.delta.ncsa.illinois.edu`
**OS:** RedHat 9 (transitioned late 2025)
**Docs:** https://docs.ncsa.illinois.edu/systems/delta/

---

## Training Strategy (Revised 2026-04-01)

**Pivot:** Qwen3.5-27B (released Feb 2026) replaces both previous base models (R1-Distill-Qwen-32B and Qwen2.5-Coder-32B-Instruct). It exceeds ALL original training targets out-of-the-box:

| Benchmark | Old Model Baseline | Original Target | Qwen3.5-27B (no training) |
|-----------|-------------------|-----------------|---------------------------|
| GPQA Diamond | 61.1% (R1-Distill-32B) | 78% | **86%** |
| LiveCodeBench | 54.6% (Qwen2.5-Coder-32B) | 65% | **80.7%** |
| AIME | 63.3% (R1-Distill-32B) | 87% | **81%** |

**Consequence:** The multi-phase training pipeline (SFT → GRPO → ETTRL) for physics and code brains is no longer needed for the general case. Training now focuses on:

1. **HEP-specific LoRA adapters** — Domain knowledge (particle physics, detector design, Geant4/ROOT/Pythia8 code patterns) that no general benchmark measures
2. **Optional general GRPO** — RL to push reasoning/coding even higher, if baseline evals on Delta confirm room for improvement worth the SU cost
3. **Router difficulty classifier retrain** — Calibrate difficulty labels against Qwen3.5-27B performance

---

## Delta Hardware Reference

### GPU Node Types We Use

| Partition | GPUs | GPU Memory | CPU | System RAM | Local SSD | SU Rate | Our Use |
|-----------|------|-----------|-----|-----------|-----------|---------|---------|
| `gpuA100x4` | 4x A100 | 40 GB HBM2 each | AMD EPYC 7763 (64-core) | 256 GB | 1.5 TB | 1 SU/GPU-hr | Primary — evals, LoRA SFT, data gen |
| `gpuA100x8` | 8x A100 | 40 GB HBM2 each | 2x AMD EPYC 7763 (128-core) | 2 TB | 1.5 TB | 1 SU/GPU-hr | Large runs (8-GPU parallelism) |
| `gpuH200x8` | 8x H200 | 141 GB HBM3 each | 2x Intel Xeon 8558 | 2 TB | 1.5 TB | 3 SU/GPU-hr | GRPO if needed (full model + ref model in memory) |

**Key constraint:** A100s are **40 GB**, not 80 GB. A 27B model in FP16 is ~54 GB — does NOT fit on a single A100. LoRA SFT requires DeepSpeed ZeRO-3 across 4x A100s or QLoRA with 4-bit base.

### File Systems

| Path | Name | Purpose | Quota | Backed Up? |
|------|------|---------|-------|-----------|
| `/u/$USER` | HOME | Scripts, configs, small files | 50 GB | Yes (daily snapshots, 14 days) |
| `/projects/<project>` | PROJECTS | Shared project data, model weights | 1 TB default | **No** |
| `/scratch/<project>` | SCRATCH | Training data, checkpoints, logs | 500 GB shared | **No** |
| `/tmp` (node-local) | LOCAL SSD | Fast temporary storage during jobs | 1.5 TB | No (wiped after job) |

**WARNING: /scratch is shared 500 GB total — don't park large models. Download right before use, clean up after.**

### Recommended File Layout

```
/u/$USER/
├── JARVIS/                    # Git repo clone (scripts + configs)
│   ├── training/
│   ├── configs/
│   └── scripts/

/projects/<project>/$USER/
├── models/                    # Pre-trained base models
│   ├── qwen3.5-27b/          # NEW — unified base
│   ├── r1-distill-qwen-32b/  # OLD — kept for reference until migration confirmed
│   └── qwen2.5-coder-32b-instruct/  # OLD — kept for reference
├── adapters/                  # Trained LoRA adapters (output)
│   ├── hep_physics/
│   └── hep_code/

/scratch/<project>/$USER/
├── data/                      # Training datasets
│   ├── hep_physics/           # HEP physics training data
│   ├── hep_code/              # HEP code training data
│   └── physics_traces/        # Phase 4A traces (old model, archive)
├── checkpoints/               # Training checkpoints
│   ├── hep_physics_sft/
│   └── hep_code_sft/
├── logs/                      # Training logs
├── eval/                      # Benchmark evaluation results
└── tb_logs/                   # TensorBoard logs
```

---

## Environment Setup

### First-time setup (run once)

```bash
# SSH into Delta
ssh login.delta.ncsa.illinois.edu

# Check your allocation
accounts   # Lists your projects and remaining SUs

# Load modules (NOT anaconda3 — it doesn't exist on Delta)
module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8

# Create venv (not conda)
python -m venv /scratch/bgde/jhill5/jarvis-venv
source /scratch/bgde/jhill5/jarvis-venv/bin/activate

# Install PyTorch (CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install training dependencies
pip install deepspeed transformers peft datasets accelerate
pip install vllm  # For inference / evals
pip install tensorboard  # Experiment tracking (no wandb)
pip install faiss-cpu  # RAG index building

# Download Qwen3.5-27B to /projects
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-27B', local_dir='/projects/bgde/jhill5/models/qwen3.5-27b')
"
```

### SLURM Job Script Template (A100)

```bash
#!/bin/bash
#SBATCH --job-name=jarvis-train
#SBATCH --account=bgde-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/bgde/jhill5/logs/%x-%j.out
#SBATCH --error=/scratch/bgde/jhill5/logs/%x-%j.err

module load python/3.13.5-gcc13.3.1
module load cudatoolkit/25.3_12.8
source /scratch/bgde/jhill5/jarvis-venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export MASTER_PORT=29500
export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache
export PYTHONUNBUFFERED=1

# Your training command here
```

---

## Revised Budget Allocation

| Phase | SUs | GPU-hrs | Partition | Purpose |
|-------|-----|---------|-----------|---------|
| 4A (old) — SPENT | ~76 | ~76 A100-hrs | gpuA100x4 | Trace generation (old model — archived) |
| 4M-eval: Baseline evals | ~20 | ~20 A100-hrs | gpuA100x4 | Confirm Qwen3.5-27B published benchmarks |
| 4A-new: HEP data curation | ~50 | ~50 A100-hrs | gpuA100x4 | Curate HEP physics + code training data |
| 4B-new: HEP physics LoRA | ~200 | ~200 A100-hrs | gpuA100x4 | QDoRA on Qwen3.5-27B, HEP physics |
| 4C-new: HEP code LoRA | ~200 | ~200 A100-hrs | gpuA100x4 | QDoRA on Qwen3.5-27B, HEP code |
| 4D-new: General GRPO | 2,000-3,000 | 2,000-3,000 | gpuA100x4/H200 | Optional — RL for general reasoning/code |
| 4E-new: Router retrain | ~50 | ~50 A100-hrs | gpuA100x4 | Difficulty classifier calibration |
| Buffer | ~4,300-5,300 | — | — | Future training, new adapters, experiments |
| **Total budget** | **8,000** | | | |
| **Spent** | **~76** | | | |

---

## Phase-by-Phase Training Commands

### Phase 4M-eval: Baseline Evals (~20 SU)

**First priority.** Confirm Qwen3.5-27B published benchmarks on our eval harnesses before committing to migration.

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=4:00:00

# GPQA Diamond — expect ~86%
python -m training.eval.run_gpqa \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --output /scratch/bgde/jhill5/eval/qwen35_gpqa.json \
  --data-dir /scratch/bgde/jhill5/data/benchmarks \
  --log-dir /scratch/bgde/jhill5/tb_logs \
  --experiment "qwen35_baseline"

# LiveCodeBench — expect ~80%
python -m training.eval.run_livecode \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --output /scratch/bgde/jhill5/eval/qwen35_livecode.json

# AIME 2024 — expect ~81%
python -m training.eval.run_aime \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --output /scratch/bgde/jhill5/eval/qwen35_aime.json
```

### Phase 4A-new: HEP Data Curation (~50 SU)

Curate HEP-specific training data from:
1. GRACE tool implementations (Geant4, ROOT, Pythia8 usage patterns)
2. HEP open-source repos (detector simulation configs, analysis scripts)
3. Existing 5,000 traces — filter for HEP-relevant problems only
4. Generate new traces from Qwen3.5-27B on HEP problems (self-distillation)

```bash
# Generate HEP-specific traces
python training/physics/generate_traces_api.py \
  --problems /scratch/bgde/jhill5/data/hep_problems.jsonl \
  --output /scratch/bgde/jhill5/data/hep_physics/ \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --traces_per_problem 8

# Filter and format for SFT
python training/physics/rejection_sample.py \
  --traces /scratch/bgde/jhill5/data/hep_physics/ \
  --output /scratch/bgde/jhill5/data/hep_physics_filtered.jsonl
```

### Phase 4B-new: HEP Physics LoRA (~200 SU)

QDoRA (rank-32) on Qwen3.5-27B with HEP physics data.

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=24:00:00

deepspeed --num_gpus=4 --module training.physics.run_sft \
  --model_name_or_path /projects/bgde/jhill5/models/qwen3.5-27b \
  --train_data /scratch/bgde/jhill5/data/hep_physics_filtered.jsonl \
  --output_dir /scratch/bgde/jhill5/checkpoints/hep_physics_sft/ \
  --deepspeed configs/ds_zero3.json \
  --use_dora true \
  --lora_rank 32 --lora_alpha 64 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --learning_rate 2e-5 --warmup_ratio 0.03 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --max_seq_length 8192 --bf16 \
  --save_strategy epoch \
  --logging_steps 10 \
  --logging_dir /scratch/bgde/jhill5/tb_logs/hep_physics_sft
```

### Phase 4C-new: HEP Code LoRA (~200 SU)

Same as above but with HEP code data (Geant4, ROOT, Pythia8 patterns).

```bash
deepspeed --num_gpus=4 --module training.physics.run_sft \
  --model_name_or_path /projects/bgde/jhill5/models/qwen3.5-27b \
  --train_data /scratch/bgde/jhill5/data/hep_code_filtered.jsonl \
  --output_dir /scratch/bgde/jhill5/checkpoints/hep_code_sft/ \
  --deepspeed configs/ds_zero3.json \
  --use_dora true \
  --lora_rank 32 --lora_alpha 64 \
  --learning_rate 2e-5 --warmup_ratio 0.03 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --max_seq_length 8192 --bf16 \
  --save_strategy epoch \
  --logging_dir /scratch/bgde/jhill5/tb_logs/hep_code_sft
```

### Phase 4D-new: General GRPO (Optional, ~2,000-3,000 SU)

Only pursue if baseline evals confirm significant room for improvement. RL with verifiable rewards (math: correct answer, code: execution pass, physics: ground truth).

```bash
#SBATCH --partition=gpuA100x4 (or gpuH200x8 if model+ref doesn't fit)
#SBATCH --gpus-per-node=4 --time=72:00:00

python -m verl.trainer.main_ppo \
  --config configs/qwen35_grpo.yaml \
  --model.path /projects/bgde/jhill5/models/qwen3.5-27b \
  --trainer.total_steps 500 \
  --trainer.save_dir /scratch/bgde/jhill5/checkpoints/qwen35_grpo/ \
  --trainer.checkpoint_interval 100
```

---

## Evaluation

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=4:00:00

# Physics: GPQA Diamond
python -m training.eval.run_gpqa \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --adapter /projects/bgde/jhill5/adapters/hep_physics/ \
  --output /scratch/bgde/jhill5/eval/hep_physics_gpqa.json

# Code: LiveCodeBench
python -m training.eval.run_livecode \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --output /scratch/bgde/jhill5/eval/qwen35_livecode.json

# Math: AIME 2024
python -m training.eval.run_aime \
  --model /projects/bgde/jhill5/models/qwen3.5-27b \
  --output /scratch/bgde/jhill5/eval/qwen35_aime.json
```

---

## Export to DGX Spark

```bash
# Package adapters for transfer
tar czf jarvis_hep_adapters.tar.gz \
  -C /projects/bgde/jhill5 \
  adapters/hep_physics/ \
  adapters/hep_code/

# Transfer via Globus ("NCSA Delta" endpoint) or scp to DGX Spark
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM on A100 40GB during SFT | Enable `offload_optimizer: {device: cpu}` in ZeRO-3 config. Reduce batch size. Enable gradient checkpointing. |
| Qwen3.5-27B doesn't fit on single A100 | 27B at FP16 = ~54 GB > 40 GB. Must use ZeRO-3 across 4 GPUs or QLoRA with 4-bit base. |
| vLLM doesn't support Qwen3.5 | Check vLLM version. Qwen3.5 support added in vLLM 0.18+. Update with `pip install -U vllm`. |
| Job preempted | Always checkpoint. Resume with `--resume_from`. |
| Slow I/O during training | Copy data to node-local `/tmp` at job start. Set `HF_HOME=/tmp/hf_cache`. |
| Check SU balance | Run `accounts` command on login node. |
| SLURM .err file has errors but .out looks fine | Always check both files. API errors and Python tracebacks go to stderr. |
| anaconda3 module not found | Use `module load python/3.13.5-gcc13.3.1` and venv instead. anaconda3 doesn't exist on Delta. |
