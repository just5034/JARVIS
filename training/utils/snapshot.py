"""Generate a structured training progress snapshot for review.

Reads TensorBoard logs and eval results to produce a concise JSON/text
summary that can be shared with Claude for analysis.

Usage:
    # Print summary to terminal (copy-paste into Claude):
    python -m training.utils.snapshot

    # Save to file:
    python -m training.utils.snapshot --output training_snapshot.json

    # Text format (easier to paste):
    python -m training.utils.snapshot --format text
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_tb_summary(log_dir: str) -> dict:
    """Extract latest metrics from TensorBoard event files."""
    summary = {}
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"_error": f"log dir not found: {log_dir}"}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        # Fallback: read meta.json files we write alongside TB events
        for exp_dir in sorted(log_path.iterdir()):
            if exp_dir.is_dir():
                meta_file = exp_dir / "meta.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    summary[exp_dir.name] = {
                        "started_at": meta.get("started_at", "?"),
                        "hparams": meta.get("hparams", {}),
                        "metrics": {},
                        "_note": "install tensorboard for full metric extraction",
                    }
        if not summary:
            summary["_error"] = "tensorboard not installed and no meta.json found"
        return summary

    for exp_dir in sorted(log_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Find the event file (could be in subdirectory)
        event_files = list(exp_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            continue

        ea = EventAccumulator(str(exp_dir))
        ea.Reload()

        exp_data = {
            "experiment": exp_dir.name,
            "metrics": {},
        }

        # Read meta.json if available
        meta_file = exp_dir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            exp_data["started_at"] = meta.get("started_at", "?")
            exp_data["hparams"] = meta.get("hparams", {})

        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                first = events[0]
                exp_data["metrics"][tag] = {
                    "latest": last.value,
                    "step": last.step,
                    "n_points": len(events),
                }
                if len(events) > 1:
                    exp_data["metrics"][tag]["first"] = first.value

        summary[exp_dir.name] = exp_data

    return summary


def get_eval_results(eval_dir: str) -> dict:
    """Read all eval result JSONs."""
    results = {}
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return {"_error": f"eval dir not found: {eval_dir}"}

    for f in sorted(eval_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            metrics = data.get("metrics", data)
            results[f.stem] = {
                "file": str(f),
                "metrics": metrics,
            }
        except Exception as e:
            results[f.stem] = {"error": str(e)}

    return results


def get_checkpoint_info(checkpoint_dir: str) -> dict:
    """Summarize available checkpoints."""
    info = {}
    cp_path = Path(checkpoint_dir)
    if not cp_path.exists():
        return {"_error": f"checkpoint dir not found: {checkpoint_dir}"}

    for d in sorted(cp_path.iterdir()):
        if d.is_dir():
            subdirs = [sd.name for sd in d.iterdir() if sd.is_dir()]
            info[d.name] = {
                "checkpoints": subdirs,
                "count": len(subdirs),
            }

    return info


def format_text(snapshot: dict) -> str:
    """Format snapshot as readable text for pasting into chat."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"JARVIS Training Snapshot — {snapshot['timestamp']}")
    lines.append("=" * 60)
    lines.append("")

    # TensorBoard experiments
    tb = snapshot.get("tb_experiments", {})
    if tb and "_error" not in tb:
        lines.append("TRAINING RUNS:")
        for exp_name, exp_data in tb.items():
            lines.append(f"  [{exp_name}]")
            if "started_at" in exp_data:
                lines.append(f"    started: {exp_data['started_at']}")
            metrics = exp_data.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                latest = metric_data.get("latest", "?")
                step = metric_data.get("step", "?")
                first = metric_data.get("first")
                delta_str = ""
                if first is not None and isinstance(latest, (int, float)) and isinstance(first, (int, float)):
                    delta = latest - first
                    delta_str = f" (delta: {delta:+.4f})"
                lines.append(f"    {metric_name}: {latest} @ step {step}{delta_str}")
            lines.append("")
    elif "_error" in tb:
        lines.append(f"TRAINING RUNS: {tb['_error']}")
        lines.append("")

    # Eval results
    evals = snapshot.get("eval_results", {})
    if evals and "_error" not in evals:
        lines.append("EVAL RESULTS:")
        targets = {
            "gpqa": ("accuracy", 0.78),
            "aime": ("accuracy", 0.87),
            "livecode": ("pass_at_1", 0.65),
        }
        for name, data in evals.items():
            metrics = data.get("metrics", {})
            lines.append(f"  [{name}]")
            for k, v in metrics.items():
                target_str = ""
                for tgt_name, (tgt_metric, tgt_val) in targets.items():
                    if tgt_name in name and k == tgt_metric:
                        status = "PASS" if isinstance(v, (int, float)) and v >= tgt_val else "FAIL"
                        target_str = f" (target: {tgt_val}, {status})"
                lines.append(f"    {k}: {v}{target_str}")
            lines.append("")
    elif "_error" in evals:
        lines.append(f"EVAL RESULTS: {evals['_error']}")
        lines.append("")

    # Checkpoints
    cps = snapshot.get("checkpoints", {})
    if cps and "_error" not in cps:
        lines.append("CHECKPOINTS:")
        for name, data in cps.items():
            lines.append(f"  {name}: {data['count']} checkpoints {data.get('checkpoints', [])}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate training progress snapshot")
    parser.add_argument(
        "--log-dir", default="/scratch/bgde/jhill5/tb_logs", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--eval-dir", default="/scratch/bgde/jhill5/eval", help="Eval results directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/scratch/bgde/jhill5/checkpoints",
        help="Checkpoints directory",
    )
    parser.add_argument("--output", default=None, help="Save snapshot to file")
    parser.add_argument(
        "--format", choices=["json", "text"], default="text", help="Output format"
    )
    args = parser.parse_args()

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tb_experiments": get_tb_summary(args.log_dir),
        "eval_results": get_eval_results(args.eval_dir),
        "checkpoints": get_checkpoint_info(args.checkpoint_dir),
    }

    if args.format == "json":
        output = json.dumps(snapshot, indent=2, default=str)
    else:
        output = format_text(snapshot)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Snapshot saved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
