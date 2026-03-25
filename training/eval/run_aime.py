"""Evaluate model on AIME 2024 — competition math problems.

Dataset: AIME 2024 I & II (30 problems total)
Metric: Accuracy (% correct, answers are integers 000-999)
Target: >= 87% for math brain (off-shelf + inference amplification)

Usage:
    python -m training.eval.run_aime \
        --model /projects/bgde-delta-gpu/models/r1-distill-qwen-32b \
        --output /scratch/bgde-delta-gpu/eval/aime_2024.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from training.eval.base import (
    extract_numeric,
    generate_batch,
    load_model,
    make_arg_parser,
)


AIME_PROMPT_TEMPLATE = """Solve the following AIME competition math problem. The answer is an integer between 000 and 999 inclusive.

Problem: {problem}

Think step by step. Show your full work, then give your final answer as a single integer."""


def load_aime_2024(data_dir: str) -> list[dict]:
    """Load AIME 2024 problems from local cache or HuggingFace."""
    local_path = Path(data_dir) / "aime" / "aime_2024.jsonl"

    if local_path.exists():
        print(f"[aime] loading from {local_path}")
        with open(local_path) as f:
            return [json.loads(line) for line in f]

    # Try HuggingFace
    print("[aime] local data not found, downloading from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    except Exception:
        try:
            ds = load_dataset("qq8933/AIME_2024", split="test")
        except Exception as e:
            print(f"ERROR: could not load AIME 2024 dataset: {e}", file=sys.stderr)
            print("Run 'python -m training.data.download_benchmarks' first.", file=sys.stderr)
            sys.exit(1)

    problems = []
    for row in ds:
        # Field names vary by dataset — try common ones
        problem_text = row.get("problem", row.get("Problem", row.get("question", "")))
        answer = str(row.get("answer", row.get("Answer", row.get("solution", ""))))

        if problem_text and answer:
            problems.append({
                "problem": problem_text,
                "answer": answer.strip(),
                "contest": row.get("contest", row.get("year", "AIME 2024")),
                "number": row.get("number", row.get("problem_number", len(problems) + 1)),
            })

    # Cache locally
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"[aime] cached {len(problems)} problems to {local_path}")

    return problems


def normalize_answer(ans: str | None) -> str | None:
    """Normalize AIME answer to 3-digit string (000-999)."""
    if ans is None:
        return None
    try:
        num = int(ans)
        if 0 <= num <= 999:
            return str(num)
    except ValueError:
        pass
    return ans


def evaluate(args) -> dict:
    """Run AIME 2024 evaluation."""
    problems = load_aime_2024(args.data_dir)
    print(f"[aime] loaded {len(problems)} problems")

    llm = load_model(args.model, args.adapter)

    prompts = [AIME_PROMPT_TEMPLATE.format(problem=p["problem"]) for p in problems]

    # Generate
    all_responses = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        responses = generate_batch(
            llm,
            batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            adapter_path=args.adapter,
        )
        all_responses.extend(responses)
        print(f"[aime] generated {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    # Score
    correct = 0
    details = []

    for problem, responses in zip(problems, all_responses):
        response_text = responses[0]
        predicted = normalize_answer(extract_numeric(response_text))
        expected = normalize_answer(problem["answer"])
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        details.append({
            "problem": problem["problem"][:200],
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
            "contest": problem.get("contest", ""),
            "number": problem.get("number", ""),
        })

    accuracy = correct / len(problems) if problems else 0
    metrics = {
        "accuracy": round(accuracy, 4),
        "n_correct": correct,
        "n_total": len(problems),
    }

    return {"metrics": metrics, "details": details}


def main():
    parser = make_arg_parser("aime_2024")
    args = parser.parse_args()

    results = evaluate(args)

    from training.utils.tracking import create_run, log_eval_results

    tracker = None
    if not args.no_track:
        tracker = create_run(
            experiment=args.experiment,
            hparams={"model": args.model, "adapter": args.adapter},
            log_dir=args.log_dir,
        )

    log_eval_results(
        tracker,
        "aime_2024",
        results["metrics"],
        details=results["details"],
        output_path=Path(args.output),
    )
    if tracker:
        tracker.close()

    target = 0.87
    acc = results["metrics"]["accuracy"]
    status = "PASS" if acc >= target else "FAIL"
    print(f"\n[aime] accuracy: {acc:.1%} (target: {target:.0%}) — {status}")


if __name__ == "__main__":
    main()
