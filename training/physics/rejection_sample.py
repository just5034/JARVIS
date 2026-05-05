"""Filter teacher-generated traces by quality via rejection sampling.

Phase 4A.3 — takes raw traces from generate_traces_api.py and filters
to keep only high-quality reasoning chains. Quality signals depend on
the domain:

  - physics: answer extraction + correctness + reasoning structure
  - code:    syntactic validity of extracted code blocks +
             reasoning-before-code structure + length sanity

Common signals:
  - Deduplication (remove near-duplicate traces per problem)
  - Length sanity (not too short, not degenerate)

Usage:
    # Physics (default; preserves prior behavior)
    python -m training.physics.rejection_sample \\
        --traces /work/hdd/bgde/jhill5/data/physics_traces/traces.jsonl \\
        --output /work/hdd/bgde/jhill5/data/hep_physics_filtered.jsonl \\
        --target-count 5000 --require-correct

    # Code (new in Phase 4C-new)
    python -m training.physics.rejection_sample \\
        --domain code \\
        --traces /work/hdd/bgde/jhill5/data/code_traces/traces.jsonl \\
        --output /work/hdd/bgde/jhill5/data/hep_code_filtered.jsonl \\
        --target-count 2500
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_answer(trace: str) -> str | None:
    """Extract the final answer from a reasoning trace.

    Multi-choice problems (GPQA, MMLU) — Qwen3.5 typically concludes with
    one of: ``**Answer:** (B)``, ``**Final Answer:** (B)``,
    ``The correct option is **(B)**``, ``matches option (A)``, or
    ``\\boxed{B}``. We extract just the letter A-D and return it
    upper-cased so the comparison against single-letter ground truth
    succeeds.

    Free-form problems — fall back to ``\\boxed{X}`` (any value) or
    ``the answer is X``. The legacy ``= X at end of line`` heuristic was
    removed because it spuriously matched equations inside the trace
    body (e.g. ``\\tau_2 = 10^{-8}`` was being returned as the answer).

    We search the trace TAIL (last ~2000 chars) because that's where the
    conclusion lives; in-body mentions of letters like ``(A)`` are noise.
    """
    tail = trace[-2000:] if len(trace) > 2000 else trace

    # Multi-choice letter patterns. Take the LAST match in the tail.
    mc_patterns = [
        r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
        r"(?i)\bfinal\s+answer[\s:.\*]+\s*\**\(?([A-D])\)?",
        r"(?i)\banswer[\s:.\*]+\s*\**\(?([A-D])\)?(?:[\s\*\.]|$)",
        r"(?i)correct\s+(?:option|answer|choice)\s+is\s+\**\(?([A-D])\)?",
        r"(?i)\bmatches\s+(?:option\s+)?\**\(?([A-D])\)?",
    ]
    best_letter = None
    best_pos = -1
    for pat in mc_patterns:
        for m in re.finditer(pat, tail):
            if m.start() > best_pos:
                best_pos = m.start()
                best_letter = m.group(1).upper()
    if best_letter:
        return best_letter

    # Free-form: \boxed{X}
    matches = list(re.finditer(r"\\boxed\{([^}]*)\}", trace))
    if matches:
        return matches[-1].group(1).strip()

    # Free-form: "the answer is X"
    match = re.search(r"[Tt]he (?:final )?answer is[:\s]+(.+?)[\.\n]", trace)
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


# ---------------------------------------------------------------------------
# Code-domain quality
# ---------------------------------------------------------------------------

# Recognise fenced blocks: ```python, ```cpp, ```cmake, plus generic ``` blocks.
_CODE_BLOCK_RE = re.compile(r"```(\w+)?\n?(.*?)```", re.DOTALL)


def extract_code_blocks(trace: str) -> list[tuple[str, str]]:
    """Return a list of (language_hint, code_text) pairs from fenced blocks."""
    blocks: list[tuple[str, str]] = []
    for m in _CODE_BLOCK_RE.finditer(trace):
        lang = (m.group(1) or "").lower().strip()
        code = m.group(2) or ""
        if code.strip():
            blocks.append((lang, code))
    return blocks


def _python_compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _gdml_well_formed(code: str) -> bool:
    # GDML files start with <?xml or <gdml; teacher output may omit XML decl.
    snippet = code.strip()
    if not snippet.startswith("<"):
        return False
    try:
        # Wrap in a synthetic root if the snippet is a fragment without one.
        ET.fromstring(snippet if snippet.startswith("<?xml") else f"<root>{snippet}</root>")
        return True
    except ET.ParseError:
        return False


def _bracket_balanced(code: str) -> bool:
    """Permissive bracket balance check — works for C++/Geant4 macros."""
    pairs = {")": "(", "}": "{", "]": "["}
    stack: list[str] = []
    in_string = False
    string_char = ""
    i = 0
    while i < len(code):
        c = code[i]
        if in_string:
            if c == "\\" and i + 1 < len(code):
                i += 2
                continue
            if c == string_char:
                in_string = False
            i += 1
            continue
        if c in ('"', "'"):
            in_string = True
            string_char = c
        elif c in "({[":
            stack.append(c)
        elif c in ")}]":
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        i += 1
    return not stack


def code_block_validity(blocks: list[tuple[str, str]]) -> tuple[int, int]:
    """Return (n_blocks, n_valid_blocks) where validity is language-aware."""
    valid = 0
    for lang, code in blocks:
        if lang in ("python", "py", "python3"):
            ok = _python_compiles(code)
        elif lang in ("xml", "gdml"):
            ok = _gdml_well_formed(code)
        elif lang in ("cpp", "c++", "c", "cc", "cmake", "geant4_macro", "mac"):
            ok = _bracket_balanced(code)
        elif not lang:
            # Heuristic dispatch on content.
            if "def " in code or "import " in code:
                ok = _python_compiles(code)
            elif code.lstrip().startswith("<"):
                ok = _gdml_well_formed(code)
            else:
                ok = _bracket_balanced(code)
        else:
            # Unknown language — fall back to length sanity only.
            ok = len(code.strip()) > 20
        if ok:
            valid += 1
    return len(blocks), valid


def code_quality_score(trace: dict) -> float:
    """Score a code trace's quality (higher = better)."""
    text = trace.get("trace", "")
    score = 0.0

    # Length sanity for code: 600-8000 chars is the sweet spot.
    length = len(text)
    if length < 200:
        score -= 2.0
    elif length < 600:
        score += 0.3
    elif length <= 8000:
        score += 1.0
    else:
        score += 0.4

    # Code-block presence and validity.
    blocks = extract_code_blocks(text)
    n_blocks, n_valid = code_block_validity(blocks)
    if n_blocks == 0:
        score -= 2.0   # no code at all — almost certainly degenerate
    else:
        score += 1.0 + 0.5 * (n_valid / max(n_blocks, 1))

    # Reward reasoning *before* code (teacher should explain, then code).
    if blocks:
        first_block_pos = text.find("```")
        if first_block_pos > 200:
            score += 0.5

    # Reward identifiable function/class definitions or imports.
    structure_tokens = ["def ", "class ", "import ", "#include", "void ", "int main",
                        "<gdml", "<volume", "process Pythia"]
    for tok in structure_tokens:
        if tok in text:
            score += 0.15

    # Penalize repetition (same as physics scoring).
    lines = text.split("\n")
    unique_lines = set(line.strip() for line in lines if line.strip())
    if len(lines) > 5 and len(unique_lines) < len(lines) * 0.5:
        score -= 1.0

    return score


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

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
        "--domain",
        choices=["physics", "code"],
        default="physics",
        help="Quality model. 'physics' uses answer extraction + correctness; "
             "'code' uses code-block syntactic validity.",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.0, help="Minimum quality score to keep"
    )
    parser.add_argument(
        "--require-correct",
        action="store_true",
        help="Only keep traces with correct answers (physics domain only). "
             "Ignored when --domain=code.",
    )
    args = parser.parse_args()

    # Pick the domain-appropriate scoring function.
    score_fn = code_quality_score if args.domain == "code" else quality_score

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

    # Correctness filter (physics-domain only).
    if args.require_correct and args.domain == "physics":
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
    elif args.require_correct and args.domain == "code":
        print("[filter] --require-correct ignored for --domain=code")

    # Quality scoring
    scored = []
    for t in deduped:
        t["_quality"] = score_fn(t)
        if t["_quality"] >= args.min_quality:
            scored.append(t)
    print(f"[filter] after quality >= {args.min_quality} ({args.domain}): {len(scored)} traces")

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
