"""FastAPI application factory for JARVIS."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from jarvis import __version__
from jarvis.api.routes import router, set_state
from jarvis.brains.brain_manager import BrainManager
from jarvis.config import JarvisConfig
from jarvis.inference.engine import InferenceEngine
from jarvis.rag.retriever import PhysicsRetriever
from jarvis.router.router import Router

logger = logging.getLogger(__name__)


def create_app(
    config: JarvisConfig,
    brain_manager: BrainManager | None = None,
    query_router: Router | None = None,
    inference_engine: InferenceEngine | None = None,
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

    if inference_engine is None:
        # Initialize RAG retriever for physics queries
        retriever = _init_retriever()
        inference_engine = InferenceEngine(config.inference, retriever=retriever)

    set_state(config, brain_manager, query_router, inference_engine)
    app.include_router(router)

    return app


def _init_retriever() -> PhysicsRetriever | None:
    """Try to load the physics corpus for RAG. Returns None if not found."""
    # Look for corpus in common locations
    candidates = [
        Path("data/physics_corpus.json"),
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "physics_corpus.json",
    ]

    for corpus_path in candidates:
        if corpus_path.exists():
            index_path = corpus_path.with_suffix(".index")
            retriever = PhysicsRetriever(
                corpus_path=corpus_path,
                index_path=index_path if index_path.exists() else None,
            )
            retriever.load()
            if retriever.loaded:
                logger.info("RAG retriever loaded (%d passages)", retriever.corpus_size)
                return retriever

    logger.info("No physics corpus found, RAG disabled")
    return None
