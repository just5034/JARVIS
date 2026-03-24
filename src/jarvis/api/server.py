"""FastAPI application factory for JARVIS."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from jarvis import __version__
from jarvis.api.routes import router, set_state
from jarvis.brains.brain_manager import BrainManager
from jarvis.config import JarvisConfig
from jarvis.router.router import Router

logger = logging.getLogger(__name__)


def create_app(
    config: JarvisConfig,
    brain_manager: BrainManager | None = None,
    query_router: Router | None = None,
) -> FastAPI:
    app = FastAPI(
        title="JARVIS",
        description="Routed multi-specialist AI system for HEP research",
        version=__version__,
    )

    if brain_manager is None:
        brain_manager = BrainManager(config)

    if query_router is None:
        query_router = Router(config)
        query_router.load()

    set_state(config, brain_manager, query_router)
    app.include_router(router)

    return app
