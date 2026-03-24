"""Tests for BrainManager and MemoryTracker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.brains.brain_manager import BrainManager
from jarvis.brains.memory_tracker import MemoryTracker
from jarvis.brains.model_loader import GenerationRequest, GenerationResult, LoadedModelHandle
from jarvis.config import JarvisConfig, MemoryBudgetConfig, load_config


@pytest.fixture
def config() -> JarvisConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    return load_config(config_dir)


# --- MemoryTracker ---


def test_memory_tracker_empty() -> None:
    budget = MemoryBudgetConfig(
        total_gb=128, reserved_os_gb=5, reserved_framework_gb=7, safety_margin_gb=5,
    )
    tracker = MemoryTracker(budget)
    assert tracker.used_gb == 0.0
    assert tracker.available_gb == 111.0


def test_memory_tracker_register() -> None:
    budget = MemoryBudgetConfig(
        total_gb=128, reserved_os_gb=5, reserved_framework_gb=7, safety_margin_gb=5,
    )
    tracker = MemoryTracker(budget)
    tracker.register("model_a", 16.0, "base")
    assert tracker.used_gb == 16.0
    assert tracker.available_gb == 95.0


def test_memory_tracker_oom() -> None:
    budget = MemoryBudgetConfig(
        total_gb=40, reserved_os_gb=2, reserved_framework_gb=4, safety_margin_gb=2,
    )
    tracker = MemoryTracker(budget)
    tracker.register("model_a", 16.0, "base")
    with pytest.raises(MemoryError, match="Cannot load"):
        tracker.register("model_b", 20.0, "base")


def test_memory_tracker_unregister() -> None:
    budget = MemoryBudgetConfig(
        total_gb=128, reserved_os_gb=5, reserved_framework_gb=7, safety_margin_gb=5,
    )
    tracker = MemoryTracker(budget)
    tracker.register("model_a", 16.0, "base")
    tracker.unregister("model_a")
    assert tracker.used_gb == 0.0


def test_memory_tracker_summary() -> None:
    budget = MemoryBudgetConfig(
        total_gb=128, reserved_os_gb=5, reserved_framework_gb=7, safety_margin_gb=5,
    )
    tracker = MemoryTracker(budget)
    tracker.register("model_a", 16.0, "base")
    summary = tracker.summary()
    assert len(summary) == 1
    assert summary[0]["name"] == "model_a"
    assert summary[0]["size_gb"] == 16.0


# --- BrainManager ---


def test_brain_manager_no_models(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    assert not mgr.has_models
    assert mgr.get_default_model() is None


def test_brain_manager_unknown_model(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    with pytest.raises(ValueError, match="Unknown model key"):
        mgr.load_base_model("nonexistent")


def test_brain_manager_load_requires_vllm(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    # On Windows/CI without vLLM, this should raise ImportError
    with pytest.raises((ImportError, Exception)):
        mgr.load_base_model("r1_distill_qwen_32b")


def test_brain_manager_with_mock_model(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    mock_handle = MagicMock(spec=LoadedModelHandle)
    mock_handle.model_key = "test"
    mgr._models["test"] = mock_handle
    mgr._default_model = "test"
    mgr.memory.register("test", 4.0, "base")

    assert mgr.has_models
    assert mgr.get_default_model() is mock_handle
    assert mgr.get_model("test") is mock_handle
    assert "test" in mgr.get_loaded_model_keys()


def test_brain_manager_unload(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    mock_handle = MagicMock(spec=LoadedModelHandle)
    mock_handle.model_key = "test"
    mgr._models["test"] = mock_handle
    mgr._default_model = "test"
    mgr.memory.register("test", 4.0, "base")

    mgr.unload_model("test")
    assert not mgr.has_models
    assert mgr.memory.used_gb == 0.0


# --- GenerationRequest / GenerationResult ---


def test_generation_request_defaults() -> None:
    req = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])
    assert req.temperature == 1.0
    assert req.max_tokens == 2048
    assert req.n == 1


def test_generation_result() -> None:
    result = GenerationResult(
        text="Hello", prompt_tokens=5, completion_tokens=1, finish_reason="stop",
    )
    assert result.text == "Hello"
    assert result.prompt_tokens == 5


# --- Adapter constraint enforcement ---


def test_swap_adapter_cross_base_rejected(config: JarvisConfig) -> None:
    """Code adapter must not load on physics base model."""
    mgr = BrainManager(config)
    mock_handle = MagicMock(spec=LoadedModelHandle)
    mgr._models["r1_distill_qwen_32b"] = mock_handle
    mgr._active_adapters["r1_distill_qwen_32b"] = None

    with pytest.raises(ValueError, match="cannot be loaded on"):
        mgr.swap_adapter("r1_distill_qwen_32b", "code_general")


def test_swap_adapter_same_base_accepted(config: JarvisConfig) -> None:
    """Physics adapter on physics base should work."""
    mgr = BrainManager(config)
    mock_handle = MagicMock(spec=LoadedModelHandle)
    mgr._models["r1_distill_qwen_32b"] = mock_handle
    mgr._active_adapters["r1_distill_qwen_32b"] = None

    mgr.swap_adapter("r1_distill_qwen_32b", "physics_general")
    assert mgr.get_active_adapter("r1_distill_qwen_32b") == "physics_general"


def test_swap_adapter_clear(config: JarvisConfig) -> None:
    """Setting adapter to None clears it."""
    mgr = BrainManager(config)
    mock_handle = MagicMock(spec=LoadedModelHandle)
    mgr._models["r1_distill_qwen_32b"] = mock_handle
    mgr._active_adapters["r1_distill_qwen_32b"] = "physics_general"

    mgr.swap_adapter("r1_distill_qwen_32b", None)
    assert mgr.get_active_adapter("r1_distill_qwen_32b") is None


def test_swap_adapter_unknown_base(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    with pytest.raises(ValueError, match="not loaded"):
        mgr.swap_adapter("nonexistent", "physics_general")


def test_swap_adapter_unknown_adapter(config: JarvisConfig) -> None:
    mgr = BrainManager(config)
    mock_handle = MagicMock(spec=LoadedModelHandle)
    mgr._models["r1_distill_qwen_32b"] = mock_handle
    mgr._active_adapters["r1_distill_qwen_32b"] = None

    with pytest.raises(ValueError, match="Unknown adapter"):
        mgr.swap_adapter("r1_distill_qwen_32b", "nonexistent_adapter")
