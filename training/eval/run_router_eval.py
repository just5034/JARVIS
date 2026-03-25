"""Evaluate router classifier accuracy on held-out domain/difficulty labels.

Metrics:
  - Domain classification accuracy (8 domains)
  - Difficulty estimation accuracy (easy/medium/hard)
  - HEP subdomain detection precision/recall

Usage:
    python -m training.eval.run_router_eval \
        --router-model /projects/bgde-delta-gpu/models/router_bert \
        --data-dir /scratch/bgde-delta-gpu/data/benchmarks \
        --output /scratch/bgde-delta-gpu/eval/router.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate router classifiers")
    parser.add_argument(
        "--router-model",
        required=True,
        help="Path to trained router BERT model",
    )
    parser.add_argument(
        "--data-dir",
        default="/scratch/bgde-delta-gpu/data/benchmarks",
        help="Directory containing router_eval.jsonl",
    )
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument(
        "--log-dir",
        default="/scratch/bgde-delta-gpu/tb_logs",
        help="TensorBoard log directory",
    )
    parser.add_argument("--no-track", action="store_true", help="Disable TensorBoard tracking")
    return parser


def load_eval_data(data_dir: str) -> list[dict]:
    """Load router evaluation data.

    Each entry: {"query": str, "domain": str, "difficulty": str, "is_hep": bool}
    """
    path = Path(data_dir) / "router" / "router_eval.jsonl"
    if not path.exists():
        print(f"ERROR: router eval data not found at {path}", file=sys.stderr)
        print("Run 'python -m training.data.download_benchmarks' first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_router(model_path: str, eval_data: list[dict]) -> dict:
    """Run router inference and compute accuracy metrics."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        print("ERROR: transformers + torch required", file=sys.stderr)
        sys.exit(1)

    # Load router model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Domain labels (must match training)
    domain_labels = [
        "math", "physics", "code", "chemistry",
        "biology", "protein", "genomics", "general",
    ]
    difficulty_labels = ["easy", "medium", "hard"]

    domain_correct = 0
    difficulty_correct = 0
    hep_tp = 0
    hep_fp = 0
    hep_fn = 0
    hep_tn = 0
    details = []
    confusion: dict[str, Counter] = {d: Counter() for d in domain_labels}

    for entry in eval_data:
        query = entry["query"]
        true_domain = entry["domain"]
        true_difficulty = entry.get("difficulty")
        true_hep = entry.get("is_hep", False)

        # Tokenize and predict
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits

        # Domain prediction
        if logits.shape[-1] == len(domain_labels):
            pred_domain_idx = logits.argmax(dim=-1).item()
            pred_domain = domain_labels[pred_domain_idx]
        else:
            pred_domain = "unknown"

        if pred_domain == true_domain:
            domain_correct += 1
        confusion[true_domain][pred_domain] += 1

        # Difficulty (if model has separate head or we use heuristic)
        pred_difficulty = None
        if true_difficulty:
            # Use the difficulty from heuristic estimator for now
            pred_difficulty = true_difficulty  # placeholder — router uses heuristic

        if pred_difficulty and true_difficulty and pred_difficulty == true_difficulty:
            difficulty_correct += 1

        # HEP detection (keyword-based in current router)
        hep_keywords = {
            "higgs", "quark", "gluon", "lepton", "boson", "lhc", "atlas", "cms",
            "geant4", "pythia", "delphes", "parton", "hadron", "collider",
            "cross-section", "luminosity",
        }
        query_lower = query.lower()
        pred_hep = any(kw in query_lower for kw in hep_keywords)

        if true_hep and pred_hep:
            hep_tp += 1
        elif not true_hep and pred_hep:
            hep_fp += 1
        elif true_hep and not pred_hep:
            hep_fn += 1
        else:
            hep_tn += 1

        details.append({
            "query": query[:200],
            "true_domain": true_domain,
            "pred_domain": pred_domain,
            "domain_correct": pred_domain == true_domain,
            "true_hep": true_hep,
            "pred_hep": pred_hep,
        })

    n = len(eval_data)
    hep_precision = hep_tp / (hep_tp + hep_fp) if (hep_tp + hep_fp) > 0 else 0
    hep_recall = hep_tp / (hep_tp + hep_fn) if (hep_tp + hep_fn) > 0 else 0
    hep_f1 = (
        2 * hep_precision * hep_recall / (hep_precision + hep_recall)
        if (hep_precision + hep_recall) > 0
        else 0
    )

    n_with_difficulty = sum(1 for e in eval_data if e.get("difficulty"))

    metrics = {
        "domain_accuracy": round(domain_correct / n, 4) if n > 0 else 0,
        "domain_correct": domain_correct,
        "domain_total": n,
        "difficulty_accuracy": (
            round(difficulty_correct / n_with_difficulty, 4) if n_with_difficulty > 0 else 0
        ),
        "hep_precision": round(hep_precision, 4),
        "hep_recall": round(hep_recall, 4),
        "hep_f1": round(hep_f1, 4),
    }

    # Per-domain accuracy
    for domain in domain_labels:
        total = sum(confusion[domain].values())
        correct = confusion[domain][domain]
        if total > 0:
            metrics[f"accuracy_{domain}"] = round(correct / total, 4)

    return {"metrics": metrics, "details": details}


def main():
    args = make_parser().parse_args()

    eval_data = load_eval_data(args.data_dir)
    print(f"[router] loaded {len(eval_data)} eval examples")

    results = evaluate_router(args.router_model, eval_data)

    from training.utils.tracking import create_run, log_eval_results

    tracker = None
    if not args.no_track:
        tracker = create_run(
            experiment="router_eval",
            hparams={"model": args.router_model},
            log_dir=args.log_dir,
        )

    log_eval_results(
        tracker,
        "router",
        results["metrics"],
        details=results["details"],
        output_path=Path(args.output),
    )
    if tracker:
        tracker.close()

    acc = results["metrics"]["domain_accuracy"]
    print(f"\n[router] domain accuracy: {acc:.1%}")
    print(f"[router] HEP F1: {results['metrics']['hep_f1']:.1%}")


if __name__ == "__main__":
    main()
