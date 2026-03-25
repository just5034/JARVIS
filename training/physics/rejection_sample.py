"""Filter teacher-generated traces by quality via rejection sampling.

Phase 4A.3 — takes raw traces from generate_traces_api.py and filters
to keep only correct, high-quality reasoning chains. Quality signals:
  1. Answer correctness (if ground truth available)
  2. Reasoning chain length (not too short, not degenerate)
  3. Deduplication (remove near-duplicate traces per problem)

Usage:
    python -m training.physics.rejection_sample \
        --traces /scratch/bgde/jhill5/data/physics_traces/traces.jsonl \
        --output /scratch/bgde/jhill5/data/physics_filtered_100k.jsonl \
        --target-count 100000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def extract_answer(trace: str) -> str | None:
    """Extract the final answer from a reasoning trace."""
    # Try boxed
    match = re.search(r"\\boxed\{([^}]*)\}", trace)
    if match:
        return match.group(1).strip()

    # Try "the answer is X"
    match = re.search(r"[Tt]he (?:final )?answer is[:\s]+(.+?)[\.\n]", trace)
    if match:
        return match.group(1).strip()

    # Try "= X" at end
    match = re.search(r"=\s*(.+?)\s*$", trace.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()

    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    ans = ans.strip().lower()
    # Remove common formatting
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", " ", ans)
    # Try numeric normalization
    try:
        val = float(ans.replace(",", ""))
        return f"{val:.6g}"
    except ValueError:
        pass
    return ans


def check_correctness(trace: dict) -> bool | None:
    """Check if trace answer matches ground truth (if available).

    Returns True/False/None (if no ground truth).
    """
    ground_truth = trace.get("answer", "")
    if not ground_truth:
        return None

    predicted = extract_answer(trace.get("trace", ""))
    if predicted is None:
        return False

    return normalize_answer(predicted) == normalize_answer(ground_truth)


def trace_hash(text: str) -> str:
    """Hash a trace for deduplication (ignoring minor whitespace differences)."""
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def quality_score(trace: dict) -> float:
    """Score a trace's quality for ranking (higher = better)."""
    text = trace.get("trace", "")
    score = 0.0

    # Reward reasonable length (500-5000 chars is good for physics)
    length = len(text)
    if length < 100:
        score -= 2.0  # too short — likely degenerate
    elif length < 500:
        score += 0.5
    elif length <= 5000:
        score += 1.0
    else:
        score += 0.5  # very long is okay but less preferred

    # Reward mathematical content
    math_indicators = [r"\\frac", r"\\int", r"\\sum", r"\\partial", "=", r"\d+\.\d+"]
    for pattern in math_indicators:
        if re.search(pattern, text):
            score += 0.2

    # Reward structured reasoning
    step_indicators = ["step", "first", "then", "therefore", "thus", "hence", "we get"]
    for word in step_indicators:
        if word in text.lower():
            score += 0.1

    # Penalize repetition
    lines = text.split("\n")
    unique_lines = set(line.strip() for line in lines if line.strip())
    if len(lines) > 5 and len(unique_lines) < len(lines) * 0.5:
        score -= 1.0  # lots of repeated lines

    # Reward having a clear final answer
    if extract_answer(text) is not None:
        score += 0.5

    return score


def main():
    parser = argparse.ArgumentParser(description="Filter traces by quality")
    parser.add_argument("--traces", required=True, help="Input traces JSONL")
    parser.add_argument("--output", required=True, help="Output filtered JSONL")
    parser.add_argument("--target-count", type=int, default=100_000, help="Target number of traces")
    parser.add_argument(
        "--min-quality", type=float, default=0.0, help="Minimum quality score to keep"
    )
    parser.add_argument(
        "--require-correct",
        action="store_true",
        help="Only keep traces with correct answers (if ground truth available)",
    )
    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: traces file not found: {traces_path}", file=sys.stderr)
        sys.exit(1)

    # Load all traces
    print(f"[filter] loading traces from {traces_path}...")
    traces = []
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    print(f"[filter] loaded {len(traces)} raw traces")

    # Deduplication
    seen_hashes: dict[str, set] = {}  # problem_id → set of trace hashes
    deduped = []
    for t in traces:
        pid = t.get("problem_id", "")
        h = trace_hash(t.get("trace", ""))
        if pid not in seen_hashes:
            seen_hashes[pid] = set()
        if h not in seen_hashes[pid]:
            seen_hashes[pid].add(h)
            deduped.append(t)
    print(f"[filter] after dedup: {len(deduped)} traces ({len(traces) - len(deduped)} duplicates)")

    # Correctness filter
    if args.require_correct:
        correct = []
        no_gt = 0
        for t in deduped:
            result = check_correctness(t)
            if result is True:
                correct.append(t)
            elif result is None:
                correct.append(t)  # keep if no ground truth
                no_gt += 1
        print(
            f"[filter] after correctness: {len(correct)} traces "
            f"({no_gt} without ground truth kept)"
        )
        deduped = correct

    # Quality scoring
    scored = []
    for t in deduped:
        t["_quality"] = quality_score(t)
        if t["_quality"] >= args.min_quality:
            scored.append(t)
    print(f"[filter] after quality >= {args.min_quality}: {len(scored)} traces")

    # Sort by quality and take top N
    scored.sort(key=lambda t: t["_quality"], reverse=True)
    selected = scored[: args.target_count]

    # Remove internal scoring field before saving
    for t in selected:
        del t["_quality"]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in selected:
            f.write(json.dumps(t) + "\n")

    # Stats
    problems = set(t.get("problem_id", "") for t in selected)
    avg_traces = len(selected) / len(problems) if problems else 0
    print(f"\n[filter] saved {len(selected)} traces for {len(problems)} problems")
    print(f"  avg traces/problem: {avg_traces:.1f}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()
