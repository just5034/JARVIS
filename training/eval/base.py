"""Base evaluator with shared model loading and scoring logic."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


def make_arg_parser(benchmark_name: str) -> argparse.ArgumentParser:
    """Create standard argument parser for eval scripts."""
    parser = argparse.ArgumentParser(description=f"Evaluate model on {benchmark_name}")
    parser.add_argument(
        "--model", required=True, help="Path to base model or HuggingFace model ID"
    )
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save results JSON (e.g., /scratch/.../eval/gpqa.json)",
    )
    parser.add_argument(
        "--log-dir",
        default="/scratch/bgde/jhill5/tb_logs",
        help="TensorBoard log directory for metric tracking",
    )
    parser.add_argument(
        "--experiment", default=benchmark_name, help="Experiment name for TensorBoard"
    )
    parser.add_argument(
        "--data-dir",
        default="/scratch/bgde/jhill5/data/benchmarks",
        help="Directory containing downloaded benchmark data",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per problem (for pass@k evaluation)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--no-track", action="store_true", help="Disable Aim tracking")
    return parser


def load_model(model_path: str, adapter_path: str | None = None):
    """Load a model via vLLM for fast batched inference.

    Automatically detects available GPUs and uses tensor parallelism
    if the model is too large for a single GPU (e.g., 32B on A100-40GB).

    Returns:
        vllm.LLM instance ready for generation.
    """
    try:
        from vllm import LLM
    except ImportError:
        print("ERROR: vllm is required for evaluation. Install with: pip install vllm", file=sys.stderr)
        sys.exit(1)

    import os
    import torch

    # Detect available GPUs for tensor parallelism
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        n_gpus = torch.cuda.device_count()

    tp_size = max(1, n_gpus)

    kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "tensor_parallel_size": tp_size,
    }

    if adapter_path:
        kwargs["enable_lora"] = True
        print(f"[eval] loading model {model_path} with adapter {adapter_path} (TP={tp_size})")
    else:
        print(f"[eval] loading model {model_path} (TP={tp_size})")

    llm = LLM(**kwargs)
    return llm


def generate_batch(
    llm,
    prompts: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    n: int = 1,
    adapter_path: str | None = None,
) -> list[list[str]]:
    """Generate completions for a batch of prompts.

    Returns:
        List of lists — each inner list has n completions for the corresponding prompt.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )

    kwargs = {}
    if adapter_path:
        from vllm.lora.request import LoRARequest

        kwargs["lora_request"] = LoRARequest("adapter", 1, adapter_path)

    outputs = llm.generate(prompts, params, **kwargs)

    results = []
    for output in outputs:
        completions = [o.text for o in output.outputs]
        results.append(completions)
    return results


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract answer from \boxed{...} in model output."""
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_choice(text: str) -> str | None:
    """Extract multiple choice answer (A/B/C/D) from model output."""
    # Look for explicit "The answer is (X)" patterns first
    patterns = [
        r"[Tt]he\s+answer\s+is\s*[:\s]*\(?([A-D])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-D])\)?",
        r"\b([A-D])\s*\)?\s*$",  # letter at end of response
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # Fallback: find the last standalone A/B/C/D
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()

    return None


def extract_numeric(text: str) -> str | None:
    """Extract a numeric answer from model output (for AIME-style problems)."""
    # Look for boxed answer first
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # Look for "the answer is X" patterns
    match = re.search(r"[Tt]he\s+answer\s+is\s*[:\s]*(\d+)", text)
    if match:
        return match.group(1)

    # Look for "= X" near the end
    match = re.search(r"=\s*(\d+)\s*$", text.strip())
    if match:
        return match.group(1)

    # Fallback: last integer in the text
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        return matches[-1]

    return None
