"""Shared test fixtures for JARVIS."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.brains.model_loader import GenerationRequest, GenerationResult
from jarvis.config import JarvisConfig, load_config


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def jarvis_config(config_dir: Path) -> JarvisConfig:
    return load_config(config_dir)


class MockModelHandle:
    """Mock model handle that returns configurable responses for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.model_key = "mock_model"
        self.model_id = "mock/model"
        self._responses = responses or ["Mock response"]
        self._call_count = 0

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        self._call_count += 1
        n = request.n
        results = []
        for i in range(n):
            idx = i % len(self._responses)
            results.append(
                GenerationResult(
                    text=self._responses[idx],
                    prompt_tokens=10,
                    completion_tokens=20,
                    finish_reason="stop",
                )
            )
        return results

    def generate_stream(self, request: GenerationRequest):
        results = self.generate(request)
        yield from results


@pytest.fixture
def mock_model() -> MockModelHandle:
    return MockModelHandle()
