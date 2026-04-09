"""Run SWE-bench Verified eval against a vLLM endpoint serving Qwen3.5-27B.

Two-step pipeline:
1. THIS SCRIPT: Generate patches for each instance using the agent loop.
   Output: predictions.json in SWE-bench format.
2. SCORING (separate, requires Docker): Run the official swebench harness
   to score predictions against the test suites.

Usage:
    python -m training.eval.run_swebench \\
        --base-url http://localhost:8000/v1 \\
        --model /projects/bgde/jhill5/models/qwen3.5-27b \\
        --output /scratch/bgde/jhill5/eval/swebench_predictions.json \\
        --workdir /scratch/bgde/jhill5/swebench_workspaces \\
        --n-instances 20

Setup notes:
- Each instance clones a real GitHub repo at a specific commit. Needs ~5-50MB
  per repo. Workdir should be on /scratch (not /tmp) to persist across runs.
- Docker is NOT required for prediction generation (only for scoring).
- Set --n-instances to a small number (10-20) for first runs to validate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_swebench_verified(n_instances: int | None = None) -> list[dict]:
    """Load SWE-bench Verified from HuggingFace.

    Returns list of instance dicts with at least:
        instance_id, repo, base_commit, problem_statement, FAIL_TO_PASS, PASS_TO_PASS
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets package required. pip install datasets", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading princeton-nlp/SWE-bench_Verified from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = list(ds)
    logger.info(f"Loaded {len(instances)} instances")

    if n_instances:
        instances = instances[:n_instances]
        logger.info(f"Subsampled to first {n_instances} instances")

    return instances


def setup_repo(instance: dict, workdir: Path) -> Path | None:
    """Clone instance repo at base_commit. Returns path or None on failure.

    Cached: if workdir/<instance_id>/.git already exists, just resets to base_commit.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]  # e.g., "django/django"
    base_commit = instance["base_commit"]

    repo_dir = workdir / instance_id

    if (repo_dir / ".git").exists():
        # Reset existing checkout
        try:
            subprocess.run(["git", "reset", "--hard", base_commit],
                           cwd=str(repo_dir), capture_output=True, check=True, timeout=60)
            subprocess.run(["git", "clean", "-fdx"],
                           cwd=str(repo_dir), capture_output=True, check=True, timeout=60)
            logger.info(f"  reset existing checkout to {base_commit[:8]}")
            return repo_dir
        except subprocess.CalledProcessError as e:
            logger.warning(f"  reset failed, re-cloning: {e}")
            shutil.rmtree(repo_dir, ignore_errors=True)

    # Fresh clone
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{repo}.git"
    logger.info(f"  cloning {clone_url}...")
    try:
        subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(repo_dir)],
            check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=str(repo_dir), check=True, capture_output=True, timeout=60,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"  clone/checkout failed: {e.stderr.decode() if e.stderr else e}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"  clone/checkout timed out")
        return None

    return repo_dir


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified prediction generation")
    parser.add_argument("--base-url", required=True, help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="Model name/path served by vLLM")
    parser.add_argument("--api-key", default="not-needed", help="API key (vLLM ignores)")
    parser.add_argument("--output", required=True, help="Path to write predictions.json")
    parser.add_argument("--workdir", required=True, help="Directory for repo checkouts")
    parser.add_argument("--n-instances", type=int, default=None,
                        help="Number of instances to run (default: all)")
    parser.add_argument("--max-steps", type=int, default=25,
                        help="Max agent steps per instance")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per LLM call")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N instances (for resuming)")
    args = parser.parse_args()

    # Imports that need package paths
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    from training.eval.swebench_agent import SWEBenchAgent

    # Setup
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    agent = SWEBenchAgent(
        llm_client=client,
        model=args.model,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    instances = load_swebench_verified(n_instances=args.n_instances)
    if args.start_from > 0:
        instances = instances[args.start_from:]
        logger.info(f"Starting from instance {args.start_from}")

    predictions = []
    # Resume support: load existing predictions if output exists
    if output_path.exists() and args.start_from == 0:
        try:
            with open(output_path) as f:
                predictions = json.load(f)
            logger.info(f"Loaded {len(predictions)} existing predictions from {output_path}")
        except Exception:
            predictions = []
    done_ids = {p["instance_id"] for p in predictions}

    # Stats
    stats = {"total": 0, "finished": 0, "errored": 0, "empty_patch": 0}

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        if instance_id in done_ids:
            logger.info(f"[{i+1}/{len(instances)}] {instance_id} — skipping (already done)")
            continue

        stats["total"] += 1
        logger.info(f"[{i+1}/{len(instances)}] {instance_id}")

        t0 = time.time()
        repo_dir = setup_repo(instance, workdir)
        if repo_dir is None:
            logger.error(f"  setup failed")
            stats["errored"] += 1
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": "",
                "error": "repo_setup_failed",
            })
            continue

        try:
            result = agent.solve(instance, repo_dir)
        except Exception as e:
            logger.error(f"  agent crashed: {e}")
            stats["errored"] += 1
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": args.model,
                "model_patch": "",
                "error": f"agent_crash: {e}",
            })
            continue

        elapsed = time.time() - t0

        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": args.model,
            "model_patch": result.patch,
            "n_steps": result.n_steps,
            "finished": result.finished,
            "elapsed_seconds": round(elapsed, 1),
        }
        if result.error:
            prediction["error"] = result.error
            stats["errored"] += 1
        if result.finished:
            stats["finished"] += 1
        if not result.patch.strip():
            stats["empty_patch"] += 1

        predictions.append(prediction)
        logger.info(f"  done in {elapsed:.0f}s, {result.n_steps} steps, "
                    f"finished={result.finished}, patch={len(result.patch)} chars")

        # Incremental save after each instance
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    # Final summary
    logger.info("=" * 60)
    logger.info("SWE-BENCH PREDICTION GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed:  {stats['total']}")
    logger.info(f"Finished cleanly: {stats['finished']}")
    logger.info(f"Errored:          {stats['errored']}")
    logger.info(f"Empty patches:    {stats['empty_patch']}")
    logger.info(f"Predictions saved to: {output_path}")
    logger.info("")
    logger.info("Next step: score predictions with the official swebench harness:")
    logger.info("  pip install swebench")
    logger.info("  python -m swebench.harness.run_evaluation \\")
    logger.info(f"      --predictions_path {output_path} \\")
    logger.info("      --max_workers 4 \\")
    logger.info("      --run_id qwen35_baseline \\")
    logger.info("      --dataset_name princeton-nlp/SWE-bench_Verified")
    logger.info("(Scoring requires Docker — run on a machine that has it.)")


if __name__ == "__main__":
    main()
