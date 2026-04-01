# JARVIS Deployment Guide

**Target Hardware:** NVIDIA DGX Spark (Founders Edition)

---

## Hardware Specifications

| Spec | Value |
|------|-------|
| Chip | GB10 Grace Blackwell Superchip |
| CPU | 20-core ARM (10x Cortex-X925 + 10x Cortex-A725) |
| GPU | Blackwell, 6,144 CUDA cores |
| RAM | 128 GB unified LPDDR5X (shared CPU/GPU) |
| Memory Bandwidth | 273 GB/s |
| AI Compute | 1 PFLOP FP4 sparse, ~100 TFLOPS FP16 |
| Storage | 4 TB NVMe SSD |
| Networking | 2x ConnectX-7 QSFP (200Gbps), 10GbE RJ-45, Wi-Fi |
| Power | 240W max via USB-C, ~100W typical |
| OS | DGX OS (Ubuntu 24.04 based) |

## Important Limitations

- **Memory bandwidth (273 GB/s) is the primary bottleneck.** Token generation speed is limited by how fast weights can be read from memory, not by compute.
- **ARM64 architecture.** Some x86-only Python packages may not have pre-built ARM wheels. Most major ML frameworks (PyTorch, vLLM, TensorRT-LLM) have ARM support on DGX OS.
- **Thermal throttling under sustained load.** Extended inference on large models may cause thermal throttling. Monitor temperatures via `nvidia-smi` and DGX Dashboard.

---

## Initial Setup

### 1. First Boot

DGX Spark ships with DGX OS pre-installed. On first boot:
- Connect monitor (HDMI), keyboard, mouse (USB-C)
- Or run headless: connect via Ethernet, find IP via router, SSH in
- DGX Dashboard available at `http://<spark-ip>:8080`

### 2. Software Stack (Pre-installed)

The following are included with DGX OS:
- CUDA 13.0.2+ and cuDNN
- Docker with GPU passthrough
- Ollama (pre-installed for quick model testing)
- JupyterLab
- Python 3.11+

### 3. Install JARVIS Dependencies

```bash
# Clone JARVIS repo
git clone <jarvis-repo-url> ~/jarvis
cd ~/jarvis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Key packages:
# - fastapi, uvicorn          (API server)
# - vllm                      (inference engine)
# - transformers, peft        (model loading, LoRA)
# - faiss-cpu                 (RAG retrieval)
# - torch                     (PyTorch, ARM64 build)
```

### 4. Download Models

```bash
# Run the download script — fetches all models to /models/
bash scripts/download_models.sh

# This downloads:
# - Qwen3.5-27B (~14 GB at FP4)        — unified base brain
# - Qwen3.5-0.8B (~0.4 GB)             — draft model for speculative decoding (or use native MTP)
# - ThinkPRM 1.5B (~0.8 GB)            — reasoning verifier
# - ChemLLM-7B (~3.5 GB)               — chemistry specialist
# - BioMistral-7B (~3.5 GB)            — biomedical specialist
# - ESM3-open 1.4B (~0.7 GB)           — protein specialist
# - Evo 2 7B (~3.5 GB)                 — genomics specialist
# - Router classifiers (~0.06 GB)
# - HEP LoRA adapters (~0.6 GB)        — when available
#
# Total download: ~30-40 GB
# Storage on 4TB SSD: plenty of room
```

### 5. Quantize Models (if not pre-quantized)

```bash
# NVFP4 quantization using TensorRT-LLM
# DGX Spark's Blackwell GPU natively supports FP4
python scripts/quantize.py --model /models/qwen3.5-27b --format nvfp4 --output /models/qwen3.5-27b-fp4
```

---

## Memory Configuration

### Single-Base Deployment (Default)

Qwen3.5-27B handles all domains (physics, math, code, general). HEP-specific LoRA adapters hot-swap when needed. Specialists load on demand.

```yaml
# configs/deployment.yaml
deployment:
  config: "single_base"
  base_model: "qwen3.5-27b"
  adapters:
    - "hep_physics"
    - "hep_code"

memory_budget:
  total_gb: 128
  reserved_os_gb: 5
  reserved_framework_gb: 7
  available_gb: 116

  always_resident:
    qwen35_27b: 14.0              # Unified base brain
    router: 0.06
    think_prm: 0.8
    draft_model: 0.4              # Qwen3.5-0.8B (or 0 if using native MTP)
    rag_index: 5.0
    active_lora: 0.3              # One adapter loaded at a time
    total: 20.96

  available_for_specialists: 90.04  # Room for ~25 specialist 7B models simultaneously

context_window:
  kv_cache_dtype: "fp8"
  kv_quant_bits_hard: 2
  ssd_offload_enabled: true
  ssd_offload_path: "/tmp/kv_cache"
  max_model_len: 131072
```

**Memory headroom advantage:** Previous dual-base config used ~39 GB always-resident. Single Qwen3.5-27B uses ~21 GB. The extra ~18 GB enables:
- Longer context windows (131K vs 65K for medium queries)
- More parallel best-of-N candidates
- Multiple specialists loaded simultaneously
- Larger KV cache for hard problems

---

## Running JARVIS

### Start the Server

```bash
# Start JARVIS
python -m jarvis.api.server \
  --config configs/deployment.yaml \
  --host 0.0.0.0 \
  --port 8000

# Server logs to stdout and /var/log/jarvis/
# Dashboard at http://localhost:8000/health
```

### Verify It Works

```bash
# Test basic inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the cross-section for Higgs production via gluon fusion at 13 TeV?"}],
    "temperature": 0.7,
    "max_tokens": 2048
  }'

# Check loaded models
curl http://localhost:8000/v1/models

# Check memory usage
curl http://localhost:8000/admin/memory
```

### Connect GRACE

In GRACE's configuration, set the API endpoint:
```yaml
# GRACE config
llm_backend:
  base_url: "http://localhost:8000/v1"
  api_key: "not-needed"         # JARVIS doesn't require auth locally
  model: ""                      # Empty = let router decide
```

---

## Multi-Device Clustering (Optional)

### Two DGX Sparks

Follow NVIDIA's official Spark Stacking guide:

```bash
# On both nodes, configure CX7 networking
sudo wget -O /etc/netplan/40-cx7.yaml \
  https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml

# Assign IPs
# Node 1: sudo ip addr add 192.168.100.10/24 dev enP2p1s0f1np1
# Node 2: sudo ip addr add 192.168.100.11/24 dev enP2p1s0f1np1

# Distributed inference with vLLM across both nodes
# Combined memory: 256 GB — can run 405B+ models
```

### DGX Spark + Consumer GPU PC

Using EXO Labs framework for disaggregated inference:

```bash
# Requirements:
# - 10GbE connection between machines
# - EXO framework installed on both
# - CUDA-compatible GPU on second machine

# Install EXO on both machines
pip install exo-inference

# DGX Spark handles prefill (compute-heavy)
# Consumer PC handles decode (bandwidth-heavy if GPU has fast VRAM)
```

---

## Performance Expectations

| Model | Decode Speed | Time to First Token | Notes |
|-------|-------------|---------------------|-------|
| 7B specialists | ~40-50 tok/s | <1s | Good for interactive |
| Qwen3.5-27B (FP4) | ~15-20 tok/s | 3-10s | Primary brain, acceptable for GRACE |

**CES 2026 software update** delivered 2.5x improvement via TensorRT-LLM optimizations. Keep DGX OS updated for best performance.

**Note:** Qwen3.5-27B (27B params) is smaller than the previous 32B models, so decode speed should be ~15-30% faster at the same quantization level.

---

## Monitoring and Maintenance

```bash
# GPU monitoring
nvidia-smi                      # GPU utilization, memory, temperature
nvidia-smi dmon                 # Continuous monitoring

# DGX Dashboard
# Browser: http://<spark-ip>:8080

# JARVIS health
curl http://localhost:8000/health

# Logs
tail -f /var/log/jarvis/server.log
```

**Thermal management:** If sustained inference causes thermal throttling (GPU clock drops), consider:
- Reducing batch size in vLLM config
- Adding external cooling (small fan pointed at the Spark)
- Reducing the hard query timeout to limit sustained compute bursts
