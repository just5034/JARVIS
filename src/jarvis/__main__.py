"""JARVIS CLI entry point. Run with: python -m jarvis"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jarvis import __version__


def find_config_dir() -> Path:
    """Walk up from CWD looking for a configs/ directory, fall back to package root."""
    # Check common locations
    candidates = [
        Path.cwd() / "configs",
        Path(__file__).resolve().parent.parent.parent / "configs",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "configs"


def cmd_serve(args: argparse.Namespace) -> None:
    import logging

    from jarvis.config import load_config

    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, config.deployment.logging.level, logging.INFO),
        format=config.deployment.logging.format,
    )
    logger = logging.getLogger("jarvis")

    host = args.host or config.deployment.server.host
    port = args.port or config.deployment.server.port

    from jarvis.brains.brain_manager import BrainManager
    from jarvis.inference.engine import InferenceEngine
    from jarvis.router.router import Router

    brain_manager = BrainManager(config)

    # Initialize router (loads BERT classifiers if available, else uses keywords)
    query_router = Router(config)
    query_router.load()
    logger.info("Router initialized (domain + difficulty classification)")

    # Initialize inference engine
    inference_engine = InferenceEngine(config.inference)
    logger.info("Inference engine initialized (difficulty-aware amplification)")

    # Auto-load models at startup if specified
    if args.load_model:
        for model_key in args.load_model.split(","):
            model_key = model_key.strip()
            logger.info("Loading model '%s' at startup...", model_key)
            brain_manager.load_base_model(model_key, set_default=True)
            logger.info("Model '%s' ready", model_key)

    from jarvis.api.server import create_app

    app = create_app(
        config,
        brain_manager=brain_manager,
        query_router=query_router,
        inference_engine=inference_engine,
    )

    import uvicorn

    logger.info("Starting JARVIS on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


def cmd_validate(args: argparse.Namespace) -> None:
    from jarvis.config import load_config

    config = load_config(args.config)

    resident_gb = config.models.total_resident_memory_gb()
    available_gb = config.deployment.memory_budget.available_gb

    print(f"JARVIS v{__version__} — config validation passed")
    print(f"  Config dir:      {args.config}")
    print(f"  Base models:     {len(config.models.base_models)}")
    print(f"  LoRA adapters:   {len(config.models.lora_adapters)}")
    print(f"  Specialists:     {len(config.models.specialists)}")
    print(f"  Router domains:  {len(config.router.domain_to_brain)}")
    print(f"  Difficulty lvls: {len(config.inference.difficulty_levels)}")
    print(f"  Resident memory: {resident_gb:.1f} GB / {available_gb} GB available")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — Routed multi-specialist AI system for HEP research",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configs/ directory (default: auto-detect)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the JARVIS API server")
    serve_parser.add_argument("--host", type=str, default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Model key(s) to load at startup, comma-separated (e.g., 'r1_distill_qwen_32b,qwen25_coder_32b')",
    )

    # validate
    subparsers.add_parser("validate", help="Validate all config files and exit")

    args = parser.parse_args()

    if args.config is None:
        args.config = find_config_dir()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
