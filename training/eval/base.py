"""Base evaluator with shared model loading and scoring logic.

Sampling parameters match Qwen3.5 published eval methodology:
- Thinking mode: ON (default for all benchmarks)
- temperature=0.6, top_p=0.95, top_k=20 (recommended for thinking mode)
- max_tokens=32768 (our vLLM max; published uses 81920 for math/code)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# Qwen3.5 recommended sampling for thinking mode
QWEN35_THINKING_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 32768,
}


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
    parser.add_argument(
        "--max-tokens", type=int,
        default=QWEN35_THINKING_DEFAULTS["max_tokens"],
        help="Max generation tokens (published: 81920 for math/code, 32768 for MCQ)",
    )
    parser.add_argument(
        "--temperature", type=float,
        default=QWEN35_THINKING_DEFAULTS["temperature"],
        help="Sampling temperature (Qwen3.5 thinking mode: 0.6)",
    )
    parser.add_argument(
        "--top-p", type=float,
        default=QWEN35_THINKING_DEFAULTS["top_p"],
        help="Top-p sampling (Qwen3.5 thinking mode: 0.95)",
    )
    parser.add_argument(
        "--top-k", type=int,
        default=QWEN35_THINKING_DEFAULTS["top_k"],
        help="Top-k sampling (Qwen3.5 thinking mode: 20)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per problem (AIME: 4, LiveCode: 8)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--no-track", action="store_true", help="Disable tracking")
    return parser


def load_model(model_path: str, adapter_path: str | None = None):
    """Load a model via vLLM for fast batched inference."""
    try:
        from vllm import LLM
    except ImportError:
        print("ERROR: vllm is required for evaluation. Install with: pip install vllm", file=sys.stderr)
        sys.exit(1)

    import os
    import torch

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        n_gpus = torch.cuda.device_count()

    tp_size = max(1, n_gpus)

    kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": 32768,
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
    max_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    n: int = 1,
    adapter_path: str | None = None,
) -> list[list[str]]:
    """Generate completions for a batch of prompts.

    Uses Qwen3.5 thinking mode sampling defaults (temp=0.6, top_p=0.95, top_k=20).
    Thinking mode is ON by default — the model produces <think>...</think> then answer.

    Returns:
        List of lists — each inner list has n completions for the corresponding prompt.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=n,
    )

    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)

    kwargs = {}
    if adapter_path:
        from vllm.lora.request import LoRARequest
        kwargs["lora_request"] = LoRARequest("adapter", 1, adapter_path)

    outputs = llm.generate(formatted_prompts, params, **kwargs)

    results = []
    for output in outputs:
        completions = [o.text for o in output.outputs]
        results.append(completions)
    return results


def strip_thinking(text: str) -> str:
    """Extract the final answer portion from model output with thinking.

    Qwen3.5 outputs: "<think>...reasoning...</think>final answer"
    Returns text after </think>, or full text if no thinking block found.
    """
    if "</think>" in text:
        _, _, after = text.rpartition("</think>")
        after = after.strip()
        if after:
            return after
    return text


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the LAST \boxed{...} answer from model output.

    Handles nested braces like \boxed{3\cdot 5^{2}}.
    """
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[start : j - 1].strip())
        i = j if depth == 0 else idx + 1

    if matches:
        return matches[-1]
    return None


def extract_choice(text: str) -> str | None:
    """Extract multiple choice answer (A/B/C/D) from model output.

    Matches OpenAI simple-evals format: 'ANSWER: $LETTER'
    """
    # Primary: "ANSWER: X" (OpenAI simple-evals format, case-insensitive)
    match = re.search(r"(?i)ANSWER\s*:\s*\$?([A-D])\$?", text)
    if match:
        return match.group(1).upper()

    # Secondary: "The answer is (X)" patterns
    match = re.search(r"[Tt]he\s+answer\s+is\s*[:\s]*\(?([A-D])\)?", text)
    if match:
        return match.group(1).upper()

    # Tertiary: standalone letter on last line
    last_line = text.strip().split("\n")[-1]
    match = re.search(r"\b([A-D])\b", last_line)
    if match:
        return match.group(1).upper()

    return None


def extract_numeric(text: str) -> str | None:
    r"""Extract a numeric answer from model output (for AIME-style problems).

    AIME answers are integers 0-999. Extracts from \boxed{}.
    """
    boxed = extract_boxed_answer(text)
    if boxed:
        s = boxed.strip().strip("$").strip()
        if re.fullmatch(r"\d+", s):
            return s
        no_comma = s.replace(",", "")
        if re.fullmatch(r"\d+", no_comma):
            return no_comma
        nums = re.findall(r"\d+", s)
        if nums:
            return nums[-1]

    match = re.search(r"[Tt]he\s+answer\s+is\s*[:\s]*(\d+)", text)
    if match:
        return match.group(1)

    match = re.search(r"=\s*(\d+)\s*$", text.strip())
    if match:
        return match.group(1)

    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        return matches[-1]

    return None
