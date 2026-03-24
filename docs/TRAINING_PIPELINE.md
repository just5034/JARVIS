# JARVIS Training Pipeline — NCSA Delta

**Platform:** NCSA Delta (ACCESS-CI allocation)
**Budget:** 8,000 Service Units (SUs)
**Login:** `ssh login.delta.ncsa.illinois.edu`
**OS:** RedHat 9 (transitioned late 2025)
**Docs:** https://docs.ncsa.illinois.edu/systems/delta/

---

## Delta Hardware Reference

### GPU Node Types We Use

| Partition | GPUs | GPU Memory | CPU | System RAM | Local SSD | SU Rate | Our Use |
|-----------|------|-----------|-----|-----------|-----------|---------|---------|
| `gpuA100x4` | 4× A100 | 40 GB HBM2 each | AMD EPYC 7763 (64-core) | 256 GB | 1.5 TB | 1 SU/GPU-hr | Primary — SFT, GRPO, data gen |
| `gpuA100x8` | 8× A100 | 40 GB HBM2 each | 2× AMD EPYC 7763 (128-core) | 2 TB | 1.5 TB | 1 SU/GPU-hr | Large runs (8-GPU parallelism) |
| `gpuH200x8` | 8× H200 | 141 GB HBM3 each | 2× Intel Xeon 8558 | 2 TB | 1.5 TB | 3 SU/GPU-hr | ETTRL only (needs >40GB/GPU) |

**Key constraint:** A100s are **40 GB**, not 80 GB. A 32B model in FP16 is ~64 GB — it does NOT fit on a single A100. All 32B training requires either:
- DeepSpeed ZeRO-3 across 4× A100s (shards model across GPUs)
- QLoRA/QDoRA with 4-bit base model (fits on 1-2 GPUs)
- The H200 partition for phases that need full model + optimizer in memory

### File Systems

| Path | Name | Purpose | Quota | Backed Up? |
|------|------|---------|-------|-----------|
| `/u/$USER` | HOME | Scripts, configs, small files | 50 GB | Yes (daily snapshots, 14 days) |
| `/projects/<project>` | PROJECTS | Shared project data, model weights | 1 TB default (request increase) | **No** |
| `/scratch/<project>` | SCRATCH | Training data, checkpoints, logs | 50 TB default | **No** |
| `/tmp` (node-local) | LOCAL SSD | Fast temporary storage during jobs | 1.5 TB | No (wiped after job) |

**⚠️ No backups on /projects or /scratch.** You must manually back up critical checkpoints (e.g., copy to HOME or transfer off-cluster via Globus).

### Recommended File Layout

```
/u/$USER/
├── jarvis/                    # Git repo clone (scripts + configs)
│   ├── training/
│   ├── configs/
│   └── scripts/

/projects/<project>/
├── models/                    # Pre-trained base models (~100 GB)
│   ├── r1-distill-qwen-32b/
│   ├── qwen3-32b/
│   └── r1-0528/              # Teacher model (if running locally)
├── adapters/                  # Trained LoRA adapters (output)
│   ├── physics_general/
│   ├── physics_hep/
│   ├── code_general/
│   └── code_hep/

/scratch/<project>/
├── data/                      # Training datasets (~200 GB)
│   ├── physics_traces/
│   ├── ladder_curriculum/
│   ├── textbook_chapters/
│   └── code_problems/
├── checkpoints/               # Training checkpoints (~500 GB+)
│   ├── physics_sft/
│   ├── physics_grpo/
│   ├── physics_ettrl/
│   └── code_azr/
├── logs/                      # Training logs + wandb
└── eval/                      # Benchmark evaluation results
```

---

## Environment Setup

### First-time setup (run once)

```bash
# SSH into Delta
ssh login.delta.ncsa.illinois.edu

# Check your allocation
accounts   # Lists your projects and remaining SUs

# Load modules
module load PrgEnv-gnu
module load cudatoolkit/25.3_12.8

# Create conda environment
module load anaconda3
conda create -n jarvis python=3.11 -y
conda activate jarvis

# Install PyTorch (CUDA 12.x)
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia

# Install training dependencies
pip install deepspeed transformers peft datasets accelerate
pip install vllm  # For data generation / inference
pip install wandb  # Experiment tracking
pip install faiss-cpu  # RAG index building

# Clone veRL framework (for GRPO and AZR)
cd /projects/<project>
git clone https://github.com/volcengine/verl.git
cd verl && pip install -e .

# Download base models to /projects
python -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', local_dir='/projects/<project>/models/r1-distill-qwen-32b')
snapshot_download('Qwen/Qwen3-32B', local_dir='/projects/<project>/models/qwen3-32b')
"
```

### SLURM Job Script Template (A100)

```bash
#!/bin/bash
#SBATCH --job-name=jarvis-train
#SBATCH --account=<your-access-project>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G                            # Leave ~16 GB for OS
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch&projects"       # Declare file system deps
#SBATCH --output=/scratch/<project>/logs/%x-%j.out
#SBATCH --error=/scratch/<project>/logs/%x-%j.err

module load PrgEnv-gnu cudatoolkit/25.3_12.8 anaconda3
conda activate jarvis

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=jarvis
export NCCL_DEBUG=INFO
export MASTER_PORT=29500
export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache

# Your training command here
```

### SLURM Job Script Template (H200 — ETTRL only)

```bash
#!/bin/bash
#SBATCH --job-name=jarvis-ettrl
#SBATCH --account=<your-access-project>
#SBATCH --partition=gpuH200x8                 # 3× SU rate!
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4                     # Use 4 of 8 to save SUs
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch&projects"
#SBATCH --output=/scratch/<project>/logs/%x-%j.out

# Budget: 900 SU at 3×/GPU-hr = 300 H200 GPU-hrs = 4 GPUs × 75 hrs
```

---

## Budget Allocation

| Phase | SUs | GPU-hrs | Partition | Brain |
|-------|-----|---------|-----------|-------|
| A. Data generation | 350 | 350 A100-hrs | gpuA100x4 | Physics |
| B. Distillation SFT | 800 | 800 A100-hrs | gpuA100x4 | Physics |
| C. Curriculum GRPO | 2,000 | 2,000 A100-hrs | gpuA100x4 | Physics |
| D. ETTRL polish | 900 | 300 H200-hrs | gpuH200x8 | Physics |
| E. Self-distillation | 450 | 450 A100-hrs | gpuA100x4 | Physics |
| F. AZR self-play RL | 2,000 | 2,000 A100-hrs | gpuA100x4 | Code |
| G. Targeted SFT | 300 | 300 A100-hrs | gpuA100x4 | Code |
| H. Code self-distillation | 300 | 300 A100-hrs | gpuA100x4 | Code |
| I. Router + eval | 200 | 200 A100-hrs | gpuA100x4 | Router |
| J. Exploration buffer | 700 | 700 A100-hrs | gpuA100x4 | Contingency |
| **Total** | **8,000** | | | |

---

## Phase-by-Phase Training Commands

### Phase A: Data Generation (~350 SU)

**A1. Multi-teacher traces (~200 SU)**

⚠️ **R1-0528 sizing problem:** 685B MoE at 4-bit ≈ 170 GB VRAM. `gpuA100x4` has 160 GB total — doesn't fit. Options:
1. **Use DeepSeek API** (best — zero SU cost for this phase)
2. Use `gpuA100x8` partition (320 GB, but costs more SUs per hour)
3. Use R1-Distill-Qwen-32B as teacher (fits on gpuA100x4, lower quality)

```bash
# Option 1: API-based (recommended)
python training/physics/generate_traces_api.py \
  --problems /scratch/<project>/data/physics_problems.jsonl \
  --output /scratch/<project>/data/physics_traces/ \
  --model deepseek-r1-0528 \
  --traces_per_problem 8
```

**A2. LADDER curriculum (~50 SU)**

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=2 --time=12:00:00

python training/physics/ladder_curriculum.py \
  --hard_problems /scratch/<project>/data/hard_physics_5k.jsonl \
  --model /projects/<project>/models/r1-distill-qwen-32b \
  --output /scratch/<project>/data/ladder_curriculum/
```

**A3. Rejection sampling (~50 SU)**

```bash
python training/physics/rejection_sample.py \
  --traces /scratch/<project>/data/physics_traces/ \
  --output /scratch/<project>/data/physics_filtered_100k.jsonl \
  --target_count 100000
```

**A4. Synthetic textbooks (~50 SU)**

```bash
python training/physics/generate_textbooks.py \
  --output /scratch/<project>/data/textbook_chapters/ \
  --count 5000
```

### Phase B: Distillation SFT (~800 SU)

ZeRO-3 is REQUIRED — 32B model exceeds single A100 40GB.

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=48:00:00

deepspeed --num_gpus=4 training/physics/run_sft.py \
  --model_name_or_path /projects/<project>/models/r1-distill-qwen-32b \
  --train_data /scratch/<project>/data/physics_filtered_100k.jsonl \
  --output_dir /scratch/<project>/checkpoints/physics_sft/ \
  --deepspeed configs/ds_zero3.json \
  --use_dora true \
  --lora_rank 32 --lora_alpha 64 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --learning_rate 2e-5 --warmup_ratio 0.03 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --max_seq_length 8192 --bf16 \
  --save_strategy epoch
```

**DeepSpeed ZeRO-3 config** (`configs/ds_zero3.json`):
```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {"device": "cpu", "pin_memory": true}
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 4
}
```

### Phase C: Curriculum GRPO (~2,000 SU)

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=72:00:00

python -m verl.trainer.main_ppo \
  --config configs/physics_grpo.yaml \
  --model.path /scratch/<project>/checkpoints/physics_sft/final/ \
  --trainer.total_steps 500 \
  --trainer.save_dir /scratch/<project>/checkpoints/physics_grpo/ \
  --trainer.checkpoint_interval 100
```

**⚠️ Checkpoint every 100 steps.** GRPO runs are 72+ hours. Save frequently. Resume with `--resume_from`.

### Phase D: ETTRL (~900 SU on H200)

```bash
#SBATCH --partition=gpuH200x8
#SBATCH --gpus-per-node=4 --time=48:00:00

python training/physics/run_ettrl.py \
  --model /scratch/<project>/checkpoints/physics_grpo/final/ \
  --problems /scratch/<project>/data/hard_physics_500.jsonl \
  --output_dir /scratch/<project>/checkpoints/physics_ettrl/ \
  --num_solutions 64 --episodes 50 \
  --gradient_checkpointing true
```

**H200 queue tip:** Only 8 H200 nodes exist — expect queuing. Submit during off-peak (nights/weekends). Use shorter wall clock + checkpointing.

### Phase F: Code AZR (~2,000 SU)

**Validate first (200 SU):**
```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=12:00:00

python -m verl.trainer.main_azr \
  --model.path /projects/<project>/models/qwen3-32b \
  --trainer.total_steps 50 \
  --trainer.save_dir /scratch/<project>/checkpoints/azr_validation/
# CHECK: loss decreasing? proposed problems sensible? no mode collapse?
# If FAIL → switch to standard GRPO with curated datasets
```

**Full training (if validation passes):**
```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4 --time=72:00:00

python -m verl.trainer.main_azr \
  --model.path /projects/<project>/models/qwen3-32b \
  --algorithm reinforce_pp \
  --trainer.total_steps 2000 \
  --trainer.save_dir /scratch/<project>/checkpoints/code_azr/ \
  --trainer.checkpoint_interval 200
```

### Post-Training (CPU — no SUs)

```bash
# POME
python training/postprocess/pome.py \
  --base_model /projects/<project>/models/r1-distill-qwen-32b \
  --adapter /scratch/<project>/checkpoints/physics_grpo/final/ \
  --output /projects/<project>/adapters/physics_general/

# SLERP merge
python training/postprocess/slerp_merge.py \
  --checkpoints /scratch/<project>/checkpoints/physics_grpo/step_{300,400,500}/ \
  --output /projects/<project>/adapters/physics_general_merged/
```

---

## Evaluation

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=2 --time=4:00:00

# Physics: GPQA Diamond — target ≥78%
python training/eval/run_gpqa.py \
  --model /projects/<project>/models/r1-distill-qwen-32b \
  --adapter /projects/<project>/adapters/physics_general/ \
  --output /scratch/<project>/eval/physics_gpqa.json

# Code: LiveCodeBench — target ≥65%
python training/eval/run_livecode.py \
  --model /projects/<project>/models/qwen3-32b \
  --adapter /projects/<project>/adapters/code_general/ \
  --output /scratch/<project>/eval/code_livecode.json

# Math: AIME 2024 — target ≥87% (off-shelf + inference amp)
python training/eval/run_aime.py \
  --model /projects/<project>/models/r1-distill-qwen-32b \
  --output /scratch/<project>/eval/math_aime.json
```

---

## Export to DGX Spark

```bash
# Quantize for NVFP4 deployment
python training/export/quantize_adapters.py \
  --adapters /projects/<project>/adapters/ \
  --format nvfp4 \
  --output /projects/<project>/adapters_fp4/

# Package and transfer via Globus ("NCSA Delta" endpoint) or scp
tar czf jarvis_artifacts.tar.gz -C /projects/<project> adapters/ adapters_fp4/
# Transfer to DGX Spark
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM on A100 40GB during SFT | Enable `offload_optimizer: {device: cpu}` in ZeRO-3 config. Reduce batch size. Enable gradient checkpointing. |
| OOM on A100 during GRPO | GRPO needs model + ref model — very tight on 40GB. Use ZeRO-3 with CPU offload. Reduce `num_solutions` from 64 to 32. |
| ETTRL must run on H200 | Model + optimizer states exceed 40GB for 32B model. No workaround — must use H200 partition (3× cost). |
| R1-0528 doesn't fit on gpuA100x4 | 685B at 4-bit = 170 GB > 160 GB (4×40). Use API, or gpuA100x8, or substitute R1-Distill-32B as teacher. |
| Job preempted | Always checkpoint. Resume with `--resume_from`. Preempt jobs get 10 min minimum + 5 min grace. |
| H200 queue wait is days | Submit at off-peak hours. Use shorter wall clock with more frequent checkpoints. |
| Slow I/O during training | Copy data to node-local `/tmp` at job start. Set `HF_HOME=/tmp/hf_cache`. |
| Check SU balance | Run `accounts` command on login node. |
