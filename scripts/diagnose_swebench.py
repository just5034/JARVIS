#!/usr/bin/env python3
"""Diagnose SWE-bench prediction results.

Usage:
    python scripts/diagnose_swebench.py /work/hdd/bgde/jhill5/eval/swebench_qwen35_*.json
    python scripts/diagnose_swebench.py  # auto-finds latest
"""

import json
import glob
import sys
from pathlib import Path

EVAL_DIR = "/work/hdd/bgde/jhill5/eval"


def find_latest():
    files = sorted(glob.glob(f"{EVAL_DIR}/swebench_qwen35_*.json"))
    return files[-1] if files else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest()
    if not path or not Path(path).exists():
        print("No SWE-bench predictions found")
        sys.exit(1)

    print(f"File: {path}")
    with open(path) as f:
        preds = json.load(f)

    total = len(preds)
    finished = [p for p in preds if p.get("finished")]
    empty = [p for p in preds if not p.get("model_patch", "").strip()]
    errored = [p for p in preds if p.get("error")]
    has_patch = [p for p in preds if p.get("model_patch", "").strip()]

    print(f"\nTotal: {total}")
    print(f"Finished cleanly: {len(finished)}")
    print(f"Has non-empty patch: {len(has_patch)}")
    print(f"Empty patches: {len(empty)}")
    print(f"Errored: {len(errored)}")

    # Errors
    if errored:
        print(f"\n--- Errors ---")
        for p in errored:
            print(f"  {p['instance_id']}: {p.get('error','?')}")

    # Show step counts
    print(f"\n--- Steps per instance ---")
    for p in preds:
        patch_len = len(p.get("model_patch", ""))
        status = "PATCH" if patch_len > 0 else "EMPTY"
        fin = "finished" if p.get("finished") else "max_steps"
        steps = p.get("n_steps", "?")
        elapsed = p.get("elapsed_seconds", "?")
        print(f"  {p['instance_id'][:40]:40s} {steps:>3} steps  {status:5s}  {fin:10s}  {elapsed}s")

    # For empty-patch instances, check if there's a history we can inspect
    print(f"\n--- First 3 empty-patch instances (checking for agent output) ---")
    shown = 0
    for p in empty:
        if shown >= 3:
            break
        print(f"\n  Instance: {p['instance_id']}")
        print(f"  Steps: {p.get('n_steps', '?')}")
        print(f"  Error: {p.get('error', 'none')}")
        print(f"  Finished: {p.get('finished', '?')}")
        # History isn't saved to JSON (too large), but we can check fields
        keys = sorted(p.keys())
        print(f"  Fields: {keys}")
        shown += 1

    # For instances WITH patches, show first 200 chars
    if has_patch:
        print(f"\n--- Instances with patches ---")
        for p in has_patch[:3]:
            patch = p["model_patch"]
            print(f"\n  {p['instance_id']}:")
            print(f"  Steps: {p.get('n_steps','?')}, Elapsed: {p.get('elapsed_seconds','?')}s")
            print(f"  Patch ({len(patch)} chars):")
            for line in patch.split("\n")[:15]:
                print(f"    {line}")
            if patch.count("\n") > 15:
                print(f"    ... ({patch.count(chr(10))} total lines)")

    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    if len(empty) > total * 0.5:
        print("[CRITICAL] >50% empty patches. Likely causes:")
        print("  1. Model not producing <tool>...</tool> JSON blocks")
        print("     -> Check if Qwen3.5 thinking mode eats tokens before tool call")
        print("     -> May need to increase max_tokens or use /no_think for tool-use")
        print("  2. Tool calls parsed but all failed (file not found, etc.)")
        print("     -> Check SLURM .err log for tool execution errors")
        print("  3. Agent hit max_steps without calling finish()")
        print("     -> All steps were 'nudge' messages (no valid tool calls)")
    if len(errored) > 0:
        print(f"[WARN] {len(errored)} instances errored. Check error messages above.")


if __name__ == "__main__":
    main()
