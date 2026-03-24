"""Tests for the JARVIS API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.api.server import create_app
from jarvis.brains.brain_manager import BrainManager
from jarvis.config import JarvisConfig, load_config
from tests.conftest import MockModelHandle


@pytest.fixture
def config() -> JarvisConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    return load_config(config_dir)


@pytest.fixture
def client(config: JarvisConfig) -> TestClient:
    app = create_app(config)
    return TestClient(app)


@pytest.fixture
def client_with_model(config: JarvisConfig) -> TestClient:
    """Client with a mock model loaded."""
    brain_manager = BrainManager(config)
    mock = MockModelHandle(["The Higgs boson mass is approximately 125 GeV."])
    brain_manager._models["test_model"] = mock
    brain_manager._default_model = "test_model"
    brain_manager._active_adapters["test_model"] = None
    brain_manager.memory.register("test_model", 4.0, "base")
    app = create_app(config, brain_manager=brain_manager)
    return TestClient(app)


# --- Basic endpoint tests (no model needed) ---


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_models_endpoint(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    model_ids = [m["id"] for m in data["data"]]
    assert "auto" in model_ids


def test_admin_memory(client: TestClient) -> None:
    response = client.get("/admin/memory")
    assert response.status_code == 200
    data = response.json()
    assert data["total_gb"] == 128


def test_chat_completions_no_model_returns_503(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 503


# --- Chat completions with mock model ---


def test_chat_completions_success(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is the Higgs mass?"}],
            "temperature": 0.7,
            "max_tokens": 256,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "125 GeV" in data["choices"][0]["message"]["content"]


def test_chat_completions_usage_stats(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    data = response.json()
    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_jarvis_metadata(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )
    data = response.json()
    meta = data["jarvis_metadata"]
    assert meta["base_model"] == "mock_model"
    assert meta["latency_ms"] >= 0
    assert meta["inference_strategy"] is not None


def test_chat_completions_streaming(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: [DONE]" in response.text


def test_chat_completions_with_stop_string(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "\n\n",
        },
    )
    assert response.status_code == 200


def test_chat_completions_with_stop_list(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n\n", "---"],
        },
    )
    assert response.status_code == 200


# --- Admin load/unload ---


def test_admin_load_unknown_model(client: TestClient) -> None:
    response = client.post(
        "/admin/load",
        json={"model": "nonexistent_model", "action": "load"},
    )
    assert response.status_code in (404, 503)


def test_admin_load_invalid_action(client: TestClient) -> None:
    response = client.post(
        "/admin/load",
        json={"model": "r1_distill_qwen_32b", "action": "restart"},
    )
    assert response.status_code == 400


# --- Phase 2: routing metadata ---


def test_routing_metadata_in_response(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Calculate the Higgs boson decay width"}
            ],
        },
    )
    data = response.json()
    meta = data["jarvis_metadata"]
    assert meta["routed_domain"] == "physics"
    assert meta["difficulty"] in ("easy", "medium", "hard")


def test_forced_model_field(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "model": "math",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    data = response.json()
    meta = data["jarvis_metadata"]
    assert meta["routed_domain"] == "math"


def test_auto_model_field(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Write a Python sort function"}
            ],
        },
    )
    data = response.json()
    meta = data["jarvis_metadata"]
    assert meta["routed_domain"] == "code"


# --- Phase 3: inference strategy metadata ---


def test_easy_query_uses_single_pass(client_with_model: TestClient) -> None:
    response = client_with_model.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is the speed of light?"}],
        },
    )
    data = response.json()
    meta = data["jarvis_metadata"]
    assert meta["inference_strategy"] == "single_pass"
    assert meta["num_candidates"] == 1
