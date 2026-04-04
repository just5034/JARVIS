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
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max generation tokens")
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
    max_tokens: int = 4096,
    temperature: float = 0.0,
    n: int = 1,
    adapter_path: str | None = None,
) -> list[list[str]]:
    """Generate completions for a batch of prompts.

    Automatically applies the model's chat template so instruct/chat models
    receive properly formatted input (e.g., <|im_start|> tokens for Qwen2.5).

    Returns:
        List of lists — each inner list has n completions for the corresponding prompt.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )

    # Apply chat template — all JARVIS eval models are instruct/chat models
    # that expect special tokens (e.g., Qwen2.5 ChatML, Qwen3.5).
    # Note: Qwen3.5 produces "Thinking Process:...\\n</think>" by default.
    # We strip thinking blocks in each eval script via strip_thinking().
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

    Qwen3.5 outputs: "Thinking Process:...reasoning...</think>final answer"
    We try to get the text after </think>, but if that's empty or missing,
    we return the FULL text so extractors can search the entire output.
    """
    if "</think>" in text:
        _, _, after = text.rpartition("</think>")
        after = after.strip()
        if after:
            return after
    # No </think> or nothing after it — return full text for extractors
    return text


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the LAST \boxed{...} answer from model output.

    Uses the last match because thinking text may contain intermediate
    \boxed{} calculations before the final answer.
    Handles nested braces like \boxed{3\cdot 5^{2}}.
    """
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        # Find matching close brace, accounting for nesting
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

    Matches the standard OpenAI simple-evals format: 'Answer: $LETTER'
    """
    # Primary: look for "Answer: X" pattern (standard format)
    match = re.search(r"(?i)Answer\s*:\s*\$?([A-D])\$?", text)
    if match:
        return match.group(1).upper()

    # Secondary: "The answer is (X)" patterns
    match = re.search(r"[Tt]he\s+answer\s+is\s*[:\s]*\(?([A-D])\)?", text)
    if match:
        return match.group(1).upper()

    # Tertiary: letter at end of response (last line)
    last_line = text.strip().split("\n")[-1]
    match = re.search(r"\b([A-D])\b", last_line)
    if match:
        return match.group(1).upper()

    return None


def _boxed_to_int(boxed: str) -> str | None:
    """Try to resolve a \\boxed{} content to an integer string.

    Handles: plain integers ("315"), LaTeX expressions ("3\\cdot 5^{2}"),
    and comma-separated numbers ("1,234").
    """
    # Strip leading/trailing whitespace and $ signs
    s = boxed.strip().strip("$").strip()

    # Plain integer
    if re.fullmatch(r"\d+", s):
        return s

    # Comma-separated integer like "1,234"
    no_comma = s.replace(",", "")
    if re.fullmatch(r"\d+", no_comma):
        return no_comma

    # Try evaluating simple LaTeX math: replace \cdot with *, \times with *,
    # ^{n} with **n, \frac{a}{b} with (a)/(b)
    expr = s
    expr = re.sub(r"\\(?:cdot|times)", "*", expr)
    expr = re.sub(r"\^{(\d+)}", r"**\1", expr)
    expr = re.sub(r"\^(\d)", r"**\1", expr)
    expr = re.sub(r"\\frac{(\d+)}{(\d+)}", r"(\1)/(\2)", expr)
    expr = re.sub(r"[{}\\]", "", expr)  # remove remaining LaTeX
    expr = expr.strip()

    try:
        val = eval(expr)  # safe: only digits and arithmetic ops after cleaning
        if isinstance(val, (int, float)) and val == int(val):
            return str(int(val))
    except Exception:
        pass

    return None


def extract_numeric(text: str) -> str | None:
    """Extract a numeric answer from model output (for AIME-style problems)."""
    # Look for boxed answer first
    boxed = extract_boxed_answer(text)
    if boxed:
        resolved = _boxed_to_int(boxed)
        if resolved is not None:
            return resolved
        # If boxed content can't be resolved, try to find a number in it
        nums = re.findall(r"\d+", boxed)
        if nums:
            return nums[-1]

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
