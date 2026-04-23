"""Evaluate model on AIME 2024 — competition math problems.

Dataset: AIME 2024 I & II (30 problems total)
Metric: avg@N — average accuracy across N independent samples per problem
         (MathArena protocol: N=4, temp=0.6)
Target: >= 81% for Qwen3.5-27B

Usage:
    python -m training.eval.run_aime \
        --model /projects/bgde/jhill5/models/qwen3.5-27b \
        --output /work/hdd/bgde/jhill5/eval/aime_2024.json \
        --n-samples 4
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
    strip_thinking,
)


AIME_PROMPT_TEMPLATE = """Solve the following AIME competition math problem. The answer is an integer between 000 and 999 inclusive.

Problem: {problem}

Please reason step by step, and put your final answer within \\boxed{{}}."""


def load_aime_2024(data_dir: str) -> list[dict]:
    """Load AIME 2024 problems from local cache or HuggingFace."""
    local_path = Path(data_dir) / "aime" / "aime_2024.jsonl"

    if local_path.exists():
        print(f"[aime] loading from {local_path}")
        with open(local_path) as f:
            return [json.loads(line) for line in f]

    print("[aime] local data not found, downloading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    except Exception:
        try:
            ds = load_dataset("qq8933/AIME_2024", split="test")
        except Exception as e:
            print(f"ERROR: could not load AIME 2024 dataset: {e}", file=sys.stderr)
            sys.exit(1)

    problems = []
    for row in ds:
        problem_text = row.get("problem", row.get("Problem", row.get("question", "")))
        answer = str(row.get("answer", row.get("Answer", row.get("solution", ""))))

        if problem_text and answer:
            problems.append({
                "problem": problem_text,
                "answer": answer.strip(),
                "contest": row.get("contest", row.get("year", "AIME 2024")),
                "number": row.get("number", row.get("problem_number", len(problems) + 1)),
            })

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"[aime] cached {len(problems)} problems to {local_path}")

    return problems


def normalize_answer(ans: str | None) -> str | None:
    """Normalize AIME answer to string (000-999)."""
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
    """Run AIME 2024 evaluation with avg@N (MathArena protocol)."""
    problems = load_aime_2024(args.data_dir)
    print(f"[aime] loaded {len(problems)} problems")
    print(f"[aime] sampling: n={args.n_samples}, temp={args.temperature}, "
          f"top_p={args.top_p}, top_k={args.top_k}, max_tokens={args.max_tokens}")

    llm = load_model(args.model, args.adapter)

    prompts = [AIME_PROMPT_TEMPLATE.format(problem=p["problem"]) for p in problems]

    # Generate N samples per problem
    all_responses = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        responses = generate_batch(
            llm,
            batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            n=args.n_samples,
            adapter_path=args.adapter,
        )
        all_responses.extend(responses)
        print(f"[aime] generated {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    # Score: avg@N — for each problem, check each sample independently,
    # then average the per-problem pass rates
    details = []
    per_problem_rates = []

    for problem, responses in zip(problems, all_responses):
        expected = normalize_answer(problem["answer"])
        sample_results = []

        for sample_text in responses:
            after_think = strip_thinking(sample_text)
            predicted = normalize_answer(extract_numeric(after_think))
            if predicted is None:
                predicted = normalize_answer(extract_numeric(sample_text))
            sample_results.append({
                "predicted": predicted,
                "is_correct": predicted == expected,
            })

        n_correct = sum(1 for s in sample_results if s["is_correct"])
        pass_rate = n_correct / len(sample_results) if sample_results else 0
        per_problem_rates.append(pass_rate)

        detail = {
            "problem": problem["problem"][:200],
            "expected": expected,
            "n_samples": len(sample_results),
            "n_correct": n_correct,
            "pass_rate": round(pass_rate, 4),
            "predictions": [s["predicted"] for s in sample_results],
            "contest": problem.get("contest", ""),
            "number": problem.get("number", ""),
        }
        # Store response tail for problems where all samples failed
        if n_correct == 0 and responses:
            detail["response_tail"] = responses[0][-500:]
        details.append(detail)

    avg_accuracy = sum(per_problem_rates) / len(per_problem_rates) if per_problem_rates else 0
    # Also report "strict" accuracy (majority vote across N samples)
    strict_correct = sum(1 for r in per_problem_rates if r > 0.5)

    metrics = {
        f"avg_at_{args.n_samples}": round(avg_accuracy, 4),
        "strict_accuracy": round(strict_correct / len(problems), 4) if problems else 0,
        "n_total": len(problems),
        "n_samples_per_problem": args.n_samples,
    }

    return {"metrics": metrics, "details": details}


def main():
    parser = make_arg_parser("aime_2024")
    args = parser.parse_args()

    # Default to N=4 for AIME (MathArena protocol)
    if args.n_samples == 1 and "--n-samples" not in sys.argv:
        args.n_samples = 4
        print("[aime] using default n_samples=4 (MathArena avg@4 protocol)")

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

    n = args.n_samples
    avg = results["metrics"][f"avg_at_{n}"]
    strict = results["metrics"]["strict_accuracy"]
    target = 0.81
    status = "PASS" if avg >= target * 0.90 else "BELOW TARGET"
    print(f"\n[aime] avg@{n}: {avg:.1%} | strict: {strict:.1%} (target: {target:.0%}) — {status}")


if __name__ == "__main__":
    main()
