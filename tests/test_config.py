"""Tests for configuration loading and validation."""

from __future__ import annotations

from jarvis.config import JarvisConfig


def test_models_yaml_parses(jarvis_config: JarvisConfig) -> None:
    assert len(jarvis_config.models.base_models) >= 2
    assert "r1_distill_qwen_32b" in jarvis_config.models.base_models
    assert "qwen3_32b" in jarvis_config.models.base_models


def test_base_model_architectures(jarvis_config: JarvisConfig) -> None:
    physics_base = jarvis_config.models.base_models["r1_distill_qwen_32b"]
    code_base = jarvis_config.models.base_models["qwen3_32b"]
    assert physics_base.architecture == "qwen2.5"
    assert code_base.architecture == "qwen3"
    assert physics_base.architecture != code_base.architecture


def test_inference_yaml_parses(jarvis_config: JarvisConfig) -> None:
    levels = jarvis_config.inference.difficulty_levels
    assert "easy" in levels
    assert "medium" in levels
    assert "hard" in levels
    assert levels["hard"].num_candidates == 16
    assert levels["easy"].num_candidates == 1


def test_router_yaml_parses(jarvis_config: JarvisConfig) -> None:
    domains = jarvis_config.router.domain_classifier.domains
    assert "math" in domains
    assert "physics" in domains
    assert "code" in domains


def test_adapter_base_model_constraint(jarvis_config: JarvisConfig) -> None:
    adapters = jarvis_config.models.lora_adapters
    base_models = jarvis_config.models.base_models
    for name, adapter in adapters.items():
        assert adapter.base_model in base_models, (
            f"Adapter '{name}' references unknown base '{adapter.base_model}'"
        )


def test_physics_adapters_on_correct_base(jarvis_config: JarvisConfig) -> None:
    adapters = jarvis_config.models.lora_adapters
    assert adapters["physics_general"].base_model == "r1_distill_qwen_32b"
    assert adapters["physics_hep"].base_model == "r1_distill_qwen_32b"


def test_code_adapters_on_correct_base(jarvis_config: JarvisConfig) -> None:
    adapters = jarvis_config.models.lora_adapters
    assert adapters["code_general"].base_model == "qwen3_32b"
    assert adapters["code_hep"].base_model == "qwen3_32b"


def test_memory_budget_fits(jarvis_config: JarvisConfig) -> None:
    resident = jarvis_config.models.total_resident_memory_gb()
    available = jarvis_config.deployment.memory_budget.available_gb
    assert resident <= available, (
        f"Resident models ({resident:.1f} GB) exceed available memory ({available} GB)"
    )


def test_router_brain_references_valid(jarvis_config: JarvisConfig) -> None:
    for domain, mapping in jarvis_config.router.domain_to_brain.items():
        if mapping.base_model:
            assert mapping.base_model in jarvis_config.models.base_models
        if mapping.adapter:
            assert mapping.adapter in jarvis_config.models.lora_adapters
        if mapping.specialist:
            assert mapping.specialist in jarvis_config.models.specialists
