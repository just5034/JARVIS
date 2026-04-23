"""Generate physics reasoning traces from a teacher model via API.

Phase 4A — multi-teacher trace generation. Calls DeepSeek R1 (or other
strong reasoning model) to produce detailed step-by-step solutions for
physics problems. Output is used for distillation SFT in Phase 4B.

Usage:
    python -m training.physics.generate_traces_api \
        --problems /work/hdd/bgde/jhill5/data/physics_problems.jsonl \
        --output /work/hdd/bgde/jhill5/data/physics_traces/ \
        --model deepseek-reasoner \
        --api-base https://api.deepseek.com/v1 \
        --traces-per-problem 8

    # Or use a local vLLM server:
    python -m training.physics.generate_traces_api \
        --problems /work/hdd/bgde/jhill5/data/physics_problems.jsonl \
        --output /work/hdd/bgde/jhill5/data/physics_traces/ \
        --model /projects/bgde/jhill5/models/r1-distill-qwen-32b \
        --api-base http://localhost:8000/v1 \
        --traces-per-problem 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


SYSTEM_PROMPT = """You are an expert physics professor. Solve the following problem step by step.
Show all your reasoning, including intermediate calculations, physical intuition,
and any approximations you make. Be rigorous and precise."""


def generate_traces_for_problem(
    problem: dict,
    n_traces: int,
    client,
    model: str,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Generate multiple reasoning traces for a single problem."""
    traces = []
    for i in range(n_traces):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem["problem"]},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            trace_text = response.choices[0].message.content

            # Extract reasoning_content if available (DeepSeek R1 format)
            reasoning = None
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning = response.choices[0].message.reasoning_content

            traces.append({
                "problem_id": problem.get("id", ""),
                "problem": problem["problem"],
                "answer": problem.get("answer", ""),
                "domain": problem.get("domain", "physics"),
                "trace_idx": i,
                "trace": trace_text,
                "reasoning": reasoning,
                "model": model,
                "tokens": response.usage.total_tokens if response.usage else None,
            })
        except Exception as e:
            print(f"  ERROR on problem {problem.get('id', '?')} trace {i}: {e}", file=sys.stderr)

    return traces


def main():
    parser = argparse.ArgumentParser(description="Generate physics reasoning traces via API")
    parser.add_argument("--problems", required=True, help="Input JSONL of physics problems")
    parser.add_argument("--output", required=True, help="Output directory for traces")
    parser.add_argument("--model", default="deepseek-reasoner", help="Model name or ID")
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL (default: uses OPENAI_API_BASE or DeepSeek)",
    )
    parser.add_argument("--api-key", default=None, help="API key (default: OPENAI_API_KEY env)")
    parser.add_argument("--traces-per-problem", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=4, help="Parallel API calls")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated problems")
    args = parser.parse_args()

    # Setup API client
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    api_base = args.api_base or os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1")

    if not api_key:
        print("ERROR: set OPENAI_API_KEY or DEEPSEEK_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)

    # Load problems
    problems_path = Path(args.problems)
    if not problems_path.exists():
        print(f"ERROR: problems file not found: {problems_path}", file=sys.stderr)
        sys.exit(1)

    with open(problems_path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    print(f"[traces] loaded {len(problems)} problems from {problems_path}")

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "traces.jsonl"

    # Resume support
    existing_ids = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                entry = json.loads(line)
                existing_ids.add(entry.get("problem_id", ""))
        problems = [p for p in problems if p.get("id", "") not in existing_ids]
        print(f"[traces] resuming — skipping {len(existing_ids)} already-done problems")

    print(f"[traces] generating {args.traces_per_problem} traces each for {len(problems)} problems")
    print(f"[traces] model: {args.model} via {api_base}")
    print(f"[traces] workers: {args.workers}")

    total_traces = 0
    total_tokens = 0
    start_time = time.time()

    with open(output_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for problem in problems:
                future = executor.submit(
                    generate_traces_for_problem,
                    problem,
                    args.traces_per_problem,
                    client,
                    args.model,
                    args.max_tokens,
                    args.temperature,
                )
                futures[future] = problem

            for future in as_completed(futures):
                problem = futures[future]
                try:
                    traces = future.result()
                    for trace in traces:
                        out_f.write(json.dumps(trace) + "\n")
                    out_f.flush()

                    total_traces += len(traces)
                    total_tokens += sum(t.get("tokens", 0) or 0 for t in traces)

                    elapsed = time.time() - start_time
                    rate = total_traces / elapsed * 3600 if elapsed > 0 else 0
                    print(
                        f"  [{total_traces} traces, {total_tokens:,} tokens, "
                        f"{rate:.0f} traces/hr] "
                        f"problem: {problem.get('id', '?')}"
                    )
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\n[traces] done: {total_traces} traces, {total_tokens:,} tokens in {elapsed/60:.1f}min")
    print(f"[traces] saved to {output_file}")


if __name__ == "__main__":
    main()
