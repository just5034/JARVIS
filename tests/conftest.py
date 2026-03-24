"""Shared test fixtures for JARVIS."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, load_config


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def jarvis_config(config_dir: Path) -> JarvisConfig:
    return load_config(config_dir)
