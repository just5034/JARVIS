"""Evaluate model on GPQA Diamond — graduate-level physics/science MCQ.

Dataset: Idavidrein/gpqa (Diamond subset, 198 questions)
Metric: Accuracy (% correct multiple choice)
Target: >= 78% for physics brain

Usage:
    python -m training.eval.run_gpqa \
        --model /projects/bgde/jhill5/models/r1-distill-qwen-32b \
        --adapter /projects/bgde/jhill5/adapters/physics_general \
        --output /scratch/bgde/jhill5/eval/gpqa_diamond.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from training.eval.base import (
    extract_choice,
    generate_batch,
    load_model,
    make_arg_parser,
)


GPQA_PROMPT_TEMPLATE = """Answer the following graduate-level science question. Think through the problem carefully, then provide your final answer as a single letter (A, B, C, or D).

Question: {question}

(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Think step by step, then state your final answer."""


def load_gpqa_diamond(data_dir: str) -> list[dict]:
    """Load GPQA Diamond dataset from local cache or HuggingFace."""
    local_path = Path(data_dir) / "gpqa" / "gpqa_diamond.jsonl"

    if local_path.exists():
        print(f"[gpqa] loading from {local_path}")
        with open(local_path) as f:
            return [json.loads(line) for line in f]

    # Fallback: load from HuggingFace
    print("[gpqa] local data not found, downloading from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        print(f"ERROR: could not load GPQA dataset: {e}", file=sys.stderr)
        print("Run 'python -m training.data.download_benchmarks' first.", file=sys.stderr)
        sys.exit(1)

    problems = []
    for row in ds:
        # GPQA has: Question, Correct Answer, Incorrect Answer 1/2/3
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Shuffle choices and track correct index
        indices = list(range(4))
        random.seed(hash(row["Question"]))  # deterministic shuffle per question
        random.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_idx = indices.index(0)  # "Correct Answer" was at index 0
        correct_letter = "ABCD"[correct_idx]

        problems.append({
            "question": row["Question"],
            "choice_a": shuffled[0],
            "choice_b": shuffled[1],
            "choice_c": shuffled[2],
            "choice_d": shuffled[3],
            "correct": correct_letter,
            "domain": row.get("Subdomain", "unknown"),
        })

    # Cache locally
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"[gpqa] cached {len(problems)} problems to {local_path}")

    return problems


def evaluate(args) -> dict:
    """Run GPQA Diamond evaluation."""
    problems = load_gpqa_diamond(args.data_dir)
    print(f"[gpqa] loaded {len(problems)} problems")

    llm = load_model(args.model, args.adapter)

    # Build prompts
    prompts = []
    for p in problems:
        prompt = GPQA_PROMPT_TEMPLATE.format(
            question=p["question"],
            choice_a=p["choice_a"],
            choice_b=p["choice_b"],
            choice_c=p["choice_c"],
            choice_d=p["choice_d"],
        )
        prompts.append(prompt)

    # Generate in batches
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
        print(f"[gpqa] generated {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    # Score
    correct = 0
    details = []
    domain_correct: dict[str, int] = {}
    domain_total: dict[str, int] = {}

    for problem, responses in zip(problems, all_responses):
        response_text = responses[0]
        predicted = extract_choice(response_text)
        is_correct = predicted == problem["correct"]
        if is_correct:
            correct += 1

        domain = problem["domain"]
        domain_total[domain] = domain_total.get(domain, 0) + 1
        if is_correct:
            domain_correct[domain] = domain_correct.get(domain, 0) + 1

        details.append({
            "question": problem["question"][:200],
            "correct": problem["correct"],
            "predicted": predicted,
            "is_correct": is_correct,
            "domain": domain,
        })

    accuracy = correct / len(problems) if problems else 0
    metrics = {
        "accuracy": round(accuracy, 4),
        "n_correct": correct,
        "n_total": len(problems),
    }

    # Per-domain breakdown
    for domain in sorted(domain_total.keys()):
        dc = domain_correct.get(domain, 0)
        dt = domain_total[domain]
        metrics[f"accuracy_{domain}"] = round(dc / dt, 4) if dt > 0 else 0

    return {"metrics": metrics, "details": details}


def main():
    parser = make_arg_parser("gpqa_diamond")
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
        "gpqa_diamond",
        results["metrics"],
        details=results["details"],
        output_path=Path(args.output),
    )
    if tracker:
        tracker.close()

    target = 0.78
    acc = results["metrics"]["accuracy"]
    status = "PASS" if acc >= target else "FAIL"
    print(f"\n[gpqa] accuracy: {acc:.1%} (target: {target:.0%}) — {status}")


if __name__ == "__main__":
    main()
