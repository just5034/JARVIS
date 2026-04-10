"""Standalone FastAPI server for the tool-use proxy.

Runs on its own port (default 8001) so it cannot collide with the main JARVIS
API or any in-flight eval job. The upstream vLLM server is configured via
JARVIS_TOOLUSE_VLLM_URL (default http://localhost:8290).

Usage:
    python -m jarvis.tooluse.server --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import argparse
import logging

import uvicorn
from fastapi import FastAPI

from jarvis.tooluse.proxy import VLLM_URL, router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="JARVIS Tool-Use Proxy",
        description=(
            "Experimental OpenAI-compatible proxy with structured tool-call "
            "support, backed by a vLLM OpenAI server. Independent of the main "
            "JARVIS serving stack."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS tool-use proxy server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logger.info("Tool-use proxy starting on %s:%d -> upstream %s", args.host, args.port, VLLM_URL)

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
