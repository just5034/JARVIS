"""Configuration loading and validation for JARVIS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator


# --- models.yaml ---


class BaseModelConfig(BaseModel):
    model_id: str
    architecture: str
    path: str
    size_gb: float
    quantization: str
    context_length: int
    recommended_max_context: int
    load_policy: str
    roles: list[str]


class LoRAAdapterConfig(BaseModel):
    base_model: str
    path: str
    size_gb: float


class InfrastructureModelConfig(BaseModel):
    model_id: str
    path: str
    size_gb: float
    load_policy: str


class SpecialistConfig(BaseModel):
    model_id: str
    path: str
    size_gb: float
    quantization: str
    type: str
    router_domain: str
    load_policy: str
    api_adapter: str | None = None


class ModelsConfig(BaseModel):
    base_models: dict[str, BaseModelConfig]
    lora_adapters: dict[str, LoRAAdapterConfig]
    infrastructure: dict[str, InfrastructureModelConfig]
    specialists: dict[str, SpecialistConfig]

    @model_validator(mode="after")
    def validate_adapter_base_models(self) -> "ModelsConfig":
        for name, adapter in self.lora_adapters.items():
            if adapter.base_model not in self.base_models:
                raise ValueError(
                    f"LoRA adapter '{name}' references base model '{adapter.base_model}' "
                    f"which is not defined. Available: {list(self.base_models.keys())}"
                )
        return self

    def total_resident_memory_gb(self) -> float:
        total = 0.0
        for m in self.base_models.values():
            if m.load_policy == "always_resident":
                total += m.size_gb
        for m in self.infrastructure.values():
            if m.load_policy == "always_resident":
                total += m.size_gb
        return total


# --- inference.yaml ---


class DifficultyLevelConfig(BaseModel):
    strategy: str
    speculative_decoding: bool
    num_candidates: int
    thinking_budget_tokens: int
    timeout_seconds: int
    kv_cache_dtype: str
    max_context_length: int
    verification_chain: bool
    budget_forcing: bool
    budget_forcing_max_waits: int | None = None
    kv_quant_bits: int | None = None
    voting: str | None = None
    verifier: str | None = None
    selection: str | None = None


class SpeculativeDecodingConfig(BaseModel):
    draft_model: str
    max_draft_tokens: int


class CodeVerificationConfig(BaseModel):
    enabled: bool
    difficulty_threshold: str
    sandbox: str
    max_test_inputs: int
    execution_timeout_seconds: int


class ContextManagementConfig(BaseModel):
    default_kv_dtype: str
    ssd_offload_enabled: bool
    ssd_offload_path: str


class InferenceConfig(BaseModel):
    difficulty_levels: dict[str, DifficultyLevelConfig]
    speculative_decoding: SpeculativeDecodingConfig
    code_verification: CodeVerificationConfig
    context_management: ContextManagementConfig


# --- router.yaml ---


class DomainClassifierConfig(BaseModel):
    model: str
    checkpoint_path: str
    domains: list[str]
    confidence_threshold: float
    fallback_domain: str


class DifficultyEstimatorConfig(BaseModel):
    model: str
    checkpoint_path: str
    levels: list[str]
    default_level: str


class HEPSubdomainConfig(BaseModel):
    enabled: bool
    method: str
    keywords: list[str]


class BrainMapping(BaseModel):
    base_model: str | None = None
    adapter: str | None = None
    hep_adapter: str | None = None
    specialist: str | None = None


class RouterConfig(BaseModel):
    domain_classifier: DomainClassifierConfig
    difficulty_estimator: DifficultyEstimatorConfig
    hep_subdomain: HEPSubdomainConfig
    domain_to_brain: dict[str, BrainMapping]


# --- deployment.yaml ---


class ServerConfig(BaseModel):
    host: str
    port: int
    workers: int


class MemoryBudgetConfig(BaseModel):
    total_gb: int
    reserved_os_gb: int
    reserved_framework_gb: int
    safety_margin_gb: int

    @property
    def available_gb(self) -> int:
        return self.total_gb - self.reserved_os_gb - self.reserved_framework_gb - self.safety_margin_gb


class LoggingConfig(BaseModel):
    level: str
    format: str


class DeploymentConfig(BaseModel):
    server: ServerConfig
    memory_budget: MemoryBudgetConfig
    logging: LoggingConfig
    model_dir: str


# --- Top-level ---


class JarvisConfig(BaseModel):
    models: ModelsConfig
    inference: InferenceConfig
    router: RouterConfig
    deployment: DeploymentConfig

    @model_validator(mode="after")
    def validate_memory_budget(self) -> "JarvisConfig":
        resident = self.models.total_resident_memory_gb()
        available = self.deployment.memory_budget.available_gb
        if resident > available:
            raise ValueError(
                f"Always-resident models require {resident:.1f} GB but only "
                f"{available} GB available (total {self.deployment.memory_budget.total_gb} GB "
                f"minus reserves)"
            )
        return self

    @model_validator(mode="after")
    def validate_router_brain_references(self) -> "JarvisConfig":
        for domain, mapping in self.router.domain_to_brain.items():
            if mapping.base_model and mapping.base_model not in self.models.base_models:
                raise ValueError(
                    f"Router domain '{domain}' references unknown base model "
                    f"'{mapping.base_model}'"
                )
            if mapping.adapter and mapping.adapter not in self.models.lora_adapters:
                raise ValueError(
                    f"Router domain '{domain}' references unknown adapter "
                    f"'{mapping.adapter}'"
                )
            if mapping.specialist and mapping.specialist not in self.models.specialists:
                raise ValueError(
                    f"Router domain '{domain}' references unknown specialist "
                    f"'{mapping.specialist}'"
                )
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_dir(config_dir: Path) -> str:
    return os.environ.get("JARVIS_MODEL_DIR", "/models")


def load_config(config_dir: str | Path) -> JarvisConfig:
    """Load and validate all JARVIS configuration files."""
    config_dir = Path(config_dir)

    deployment_data = _load_yaml(config_dir / "deployment.yaml")

    model_dir_override = os.environ.get("JARVIS_MODEL_DIR")
    if model_dir_override:
        deployment_data["model_dir"] = model_dir_override

    return JarvisConfig(
        models=ModelsConfig(**_load_yaml(config_dir / "models.yaml")),
        inference=InferenceConfig(**_load_yaml(config_dir / "inference.yaml")),
        router=RouterConfig(**_load_yaml(config_dir / "router.yaml")),
        deployment=DeploymentConfig(**deployment_data),
    )
