"""LADDER curriculum generation — decompose hard problems into scaffolded sequences.

Phase 4A.2 — for each hard physics problem, generate a sequence of increasingly
difficult sub-problems that build toward the full solution. This helps the model
learn to reason incrementally rather than attempting hard problems cold.

Reference: LADDER (Learning through Ascending Difficulty for Efficient Reasoning)

Usage:
    python -m training.physics.ladder_curriculum \
        --hard-problems /scratch/bgde/jhill5/data/hard_physics_5k.jsonl \
        --model /projects/bgde/jhill5/models/r1-distill-qwen-32b \
        --output /scratch/bgde/jhill5/data/ladder_curriculum/ \
        --api-base http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DECOMPOSE_PROMPT = """You are an expert physics professor creating a scaffolded learning sequence.

Given this HARD problem:
{problem}

Create a sequence of 3-5 sub-problems that build toward solving the full problem.
Each sub-problem should:
1. Be self-contained and solvable independently
2. Increase in difficulty
3. Build on concepts from previous sub-problems
4. The final sub-problem should be close to the original hard problem

Format your response as a JSON array of objects with "difficulty" (1-5), "problem", and "hint" fields.
Return ONLY the JSON array, no other text."""

SOLVE_PROMPT = """Solve the following physics problem step by step. Show all work.

{problem}

{hint}"""


def decompose_problem(client, model: str, problem: dict, max_tokens: int) -> list[dict]:
    """Use LLM to decompose a hard problem into a scaffolded sequence."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": DECOMPOSE_PROMPT.format(problem=problem["problem"]),
            }
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()

    # Parse JSON from response
    # Handle case where model wraps in markdown code block
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        sub_problems = json.loads(text)
        if isinstance(sub_problems, list):
            return sub_problems
    except json.JSONDecodeError:
        pass

    return []


def generate_trace_for_subproblem(
    client, model: str, sub_problem: dict, max_tokens: int
) -> str:
    """Generate a reasoning trace for a sub-problem."""
    hint_text = f"Hint: {sub_problem['hint']}" if sub_problem.get("hint") else ""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": SOLVE_PROMPT.format(
                    problem=sub_problem["problem"], hint=hint_text
                ),
            }
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Generate LADDER curriculum")
    parser.add_argument("--hard-problems", required=True, help="Input hard problems JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True, help="Model for decomposition + solving")
    parser.add_argument("--api-base", default=None, help="API base URL for vLLM server")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N problems")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    api_base = args.api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    client = OpenAI(api_key=api_key, base_url=api_base)

    # Load hard problems
    with open(args.hard_problems) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        problems = problems[: args.limit]
    print(f"[ladder] loaded {len(problems)} hard problems")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    curriculum_file = output_dir / "curriculum.jsonl"

    total_sequences = 0
    total_sub_problems = 0

    with open(curriculum_file, "w") as out_f:
        for i, problem in enumerate(problems):
            print(f"[ladder] problem {i + 1}/{len(problems)}: {problem.get('id', '?')}")

            # Decompose into sub-problems
            sub_problems = decompose_problem(client, args.model, problem, args.max_tokens)
            if not sub_problems:
                print(f"  WARNING: failed to decompose, skipping")
                continue

            # Generate trace for each sub-problem
            sequence = {
                "original_problem_id": problem.get("id", ""),
                "original_problem": problem["problem"],
                "original_answer": problem.get("answer", ""),
                "sub_problems": [],
            }

            for j, sp in enumerate(sub_problems):
                try:
                    trace = generate_trace_for_subproblem(
                        client, args.model, sp, args.max_tokens
                    )
                    sequence["sub_problems"].append({
                        "difficulty": sp.get("difficulty", j + 1),
                        "problem": sp["problem"],
                        "hint": sp.get("hint", ""),
                        "trace": trace,
                    })
                except Exception as e:
                    print(f"  WARNING: failed sub-problem {j}: {e}")

            if sequence["sub_problems"]:
                out_f.write(json.dumps(sequence) + "\n")
                out_f.flush()
                total_sequences += 1
                total_sub_problems += len(sequence["sub_problems"])

            print(f"  → {len(sequence['sub_problems'])} sub-problems generated")

    print(f"\n[ladder] done: {total_sequences} sequences, {total_sub_problems} sub-problems")
    print(f"[ladder] saved to {curriculum_file}")


if __name__ == "__main__":
    main()
