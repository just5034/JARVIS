"""Standalone FastAPI server for the tool-use proxy and Anthropic shim.

Serves two roles depending on how it's launched:
  --mode openai   (default) OpenAI-compatible proxy on port 8001
  --mode anthropic         Anthropic Messages API shim on port 8000
  --mode both              Both routers on a single port (8000)

The OpenAI proxy forwards tool-bearing requests to vLLM (port 8290).
The Anthropic shim translates Anthropic Messages format to/from OpenAI,
so CodeAgent can connect with:
    ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=dummy

Usage:
    python -m jarvis.tooluse.server --mode openai --port 8001
    python -m jarvis.tooluse.server --mode anthropic --port 8000
    python -m jarvis.tooluse.server --mode both --port 8000
"""

from __future__ import annotations

import argparse
import logging

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_app(mode: str = "both") -> FastAPI:
    app = FastAPI(
        title="JARVIS Tool-Use Server",
        description=(
            "OpenAI-compatible proxy with tool-call support and/or Anthropic "
            "Messages API shim for CodeAgent compatibility."
        ),
        version="0.2.0",
    )

    if mode in ("openai", "both"):
        from jarvis.tooluse.proxy import router as openai_router
        app.include_router(openai_router, tags=["OpenAI"])

    if mode in ("anthropic", "both"):
        from jarvis.tooluse.anthropic_shim import router as anthropic_router
        app.include_router(anthropic_router, tags=["Anthropic"])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS tool-use server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["openai", "anthropic", "both"], default="both")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logger.info(
        "Tool-use server starting on %s:%d (mode=%s)",
        args.host, args.port, args.mode,
    )

    uvicorn.run(create_app(args.mode), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
