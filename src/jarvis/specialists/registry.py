"""Specialist model registry — catalog of available specialist models."""

from __future__ import annotations

from jarvis.config import ModelsConfig, SpecialistConfig


class SpecialistRegistry:
    """Maintains catalog of specialist models and their metadata."""

    def __init__(self, config: ModelsConfig) -> None:
        self._specialists = config.specialists

    def get(self, domain: str) -> SpecialistConfig | None:
        return self._specialists.get(domain)

    def list_available(self) -> list[str]:
        return list(self._specialists.keys())

    def requires_adapter(self, domain: str) -> bool:
        spec = self._specialists.get(domain)
        return spec is not None and spec.api_adapter is not None
