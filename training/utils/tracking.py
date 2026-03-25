"""Aim-based experiment tracking for JARVIS training and evaluation.

Aim is a free, open-source experiment tracker (like W&B but self-hosted).
Start the dashboard with: aim up --repo /path/to/.aim
"""

from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AIM_AVAILABLE = False
try:
    from aim import Run
    from aim.hugging_face import AimCallback

    AIM_AVAILABLE = True
except ImportError:
    Run = None
    AimCallback = None

TB_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter

    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None


DEFAULT_AIM_REPO = "/scratch/bgde-delta-gpu/aim"


def create_run(
    experiment: str,
    hparams: dict[str, Any] | None = None,
    aim_repo: str = DEFAULT_AIM_REPO,
    tags: list[str] | None = None,
) -> Run | None:
    """Create an Aim run for tracking metrics.

    Args:
        experiment: Experiment name (e.g., "physics_sft", "gpqa_eval").
        hparams: Hyperparameters to log.
        aim_repo: Path to .aim repository directory.
        tags: Optional tags for filtering runs.

    Returns:
        Aim Run object, or None if Aim is not installed.
    """
    if not AIM_AVAILABLE:
        print("[tracking] aim not installed — metrics will not be tracked", file=sys.stderr)
        return None

    run = Run(repo=aim_repo, experiment=experiment)
    run["hostname"] = socket.gethostname()
    run["started_at"] = datetime.now(timezone.utc).isoformat()

    if hparams:
        run["hparams"] = hparams
    if tags:
        for tag in tags:
            run.add_tag(tag)

    return run


def get_aim_callback(
    experiment: str,
    aim_repo: str = DEFAULT_AIM_REPO,
    hparams: dict[str, Any] | None = None,
) -> AimCallback | None:
    """Get a HuggingFace Transformers callback that logs to Aim.

    Use this with `Trainer(callbacks=[get_aim_callback(...)])`.
    """
    if not AIM_AVAILABLE or AimCallback is None:
        print("[tracking] aim not installed — using TensorBoard fallback", file=sys.stderr)
        return None

    return AimCallback(repo=aim_repo, experiment=experiment)


def log_eval_results(
    run: Run | None,
    benchmark: str,
    metrics: dict[str, float],
    details: list[dict] | None = None,
    output_path: Path | None = None,
) -> None:
    """Log evaluation results to Aim and optionally save to disk.

    Args:
        run: Aim run (or None to skip tracking).
        benchmark: Benchmark name (e.g., "gpqa_diamond", "aime_2024").
        metrics: Summary metrics (e.g., {"accuracy": 0.78, "n_correct": 154}).
        details: Per-problem results for analysis.
        output_path: If provided, save full results as JSON.
    """
    if run is not None:
        for key, value in metrics.items():
            run.track(value, name=f"{benchmark}/{key}")

    results = {
        "benchmark": benchmark,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    if details:
        results["details"] = details

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"[tracking] results saved to {output_path}")

    for key, value in metrics.items():
        print(f"  {benchmark}/{key}: {value}")


class TensorBoardTracker:
    """Fallback tracker using TensorBoard when Aim is not available."""

    def __init__(self, log_dir: str):
        if not TB_AVAILABLE:
            raise RuntimeError("Neither aim nor tensorboard is installed")
        self.writer = SummaryWriter(log_dir=log_dir)

    def track(self, value: float, name: str, step: int = 0) -> None:
        self.writer.add_scalar(name, value, global_step=step)

    def close(self) -> None:
        self.writer.close()
