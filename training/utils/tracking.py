"""TensorBoard-based experiment tracking for JARVIS training and evaluation.

TensorBoard ships with PyTorch — no extra install needed.
View dashboard: tensorboard --logdir /scratch/bgde-delta-gpu/tb_logs
SSH tunnel: ssh -L 6006:localhost:6006 jhill5@login.delta.ncsa.illinois.edu
Open: http://localhost:6006
"""

from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TB_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter

    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None


DEFAULT_LOG_DIR = "/scratch/bgde-delta-gpu/tb_logs"


class TrainingTracker:
    """Unified tracker wrapping TensorBoard's SummaryWriter."""

    def __init__(self, log_dir: str, experiment: str, hparams: dict[str, Any] | None = None):
        self.experiment = experiment
        self.log_path = Path(log_dir) / experiment
        self.log_path.mkdir(parents=True, exist_ok=True)

        if not TB_AVAILABLE:
            print("[tracking] tensorboard not available — logging to JSON only", file=sys.stderr)
            self.writer = None
        else:
            self.writer = SummaryWriter(log_dir=str(self.log_path))
            if hparams:
                # Log hyperparameters as text (TensorBoard hparams plugin)
                self.writer.add_text(
                    "hparams", json.dumps(hparams, indent=2, default=str), global_step=0
                )

        # Also save metadata as JSON for the snapshot tool
        meta = {
            "experiment": experiment,
            "hostname": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hparams": hparams or {},
        }
        (self.log_path / "meta.json").write_text(json.dumps(meta, indent=2))

    def track(self, value: float, name: str, step: int = 0) -> None:
        """Log a scalar metric."""
        if self.writer:
            self.writer.add_scalar(name, value, global_step=step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()


def create_run(
    experiment: str,
    hparams: dict[str, Any] | None = None,
    log_dir: str = DEFAULT_LOG_DIR,
    **_kwargs,
) -> TrainingTracker:
    """Create a tracker for logging metrics.

    Args:
        experiment: Experiment name (e.g., "physics_sft", "gpqa_eval").
        hparams: Hyperparameters to log.
        log_dir: Root directory for TensorBoard logs.

    Returns:
        TrainingTracker instance.
    """
    return TrainingTracker(log_dir=log_dir, experiment=experiment, hparams=hparams)


def get_tensorboard_callback(log_dir: str = DEFAULT_LOG_DIR, experiment: str = "train"):
    """Get a HuggingFace Transformers TensorBoardCallback with custom log dir.

    HuggingFace Trainer has built-in TensorBoard support — just set
    `logging_dir` in TrainingArguments. This helper is for cases where
    you need explicit control.
    """
    try:
        from transformers.integrations import TensorBoardCallback

        return TensorBoardCallback()
    except ImportError:
        print("[tracking] transformers TensorBoardCallback not available", file=sys.stderr)
        return None


def log_eval_results(
    tracker: TrainingTracker | None,
    benchmark: str,
    metrics: dict[str, float],
    details: list[dict] | None = None,
    output_path: Path | None = None,
) -> None:
    """Log evaluation results to TensorBoard and save to disk.

    Args:
        tracker: TrainingTracker (or None to skip TB logging).
        benchmark: Benchmark name (e.g., "gpqa_diamond", "aime_2024").
        metrics: Summary metrics (e.g., {"accuracy": 0.78, "n_correct": 154}).
        details: Per-problem results for analysis.
        output_path: If provided, save full results as JSON.
    """
    if tracker is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tracker.track(value, name=f"{benchmark}/{key}")

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
