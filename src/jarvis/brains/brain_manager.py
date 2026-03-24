"""Brain Manager — model loading, LoRA adapter swapping, specialist routing."""

from __future__ import annotations

import logging
import time

from jarvis.brains.memory_tracker import MemoryTracker
from jarvis.brains.model_loader import LoadedModelHandle, load_model
from jarvis.config import JarvisConfig
from jarvis.router.router import RoutingDecision
from jarvis.specialists.loader import SpecialistLoader
from jarvis.specialists.registry import SpecialistRegistry

logger = logging.getLogger(__name__)


class BrainManager:
    """Manages base model loading, LoRA adapter swapping, and specialist routing."""

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self.memory = MemoryTracker(config.deployment.memory_budget)
        self._models: dict[str, LoadedModelHandle] = {}
        self._active_adapters: dict[str, str | None] = {}
        self._default_model: str | None = None
        self._specialist_registry = SpecialistRegistry(config.models)
        self._specialist_loader = SpecialistLoader(self.memory, config.deployment.model_dir)

    @property
    def has_models(self) -> bool:
        return len(self._models) > 0

    @property
    def specialist_registry(self) -> SpecialistRegistry:
        return self._specialist_registry

    @property
    def specialist_loader(self) -> SpecialistLoader:
        return self._specialist_loader

    def get_loaded_model_keys(self) -> list[str]:
        keys = list(self._models.keys())
        keys.extend(self._specialist_loader.list_loaded())
        return keys

    def get_active_adapter(self, base_key: str) -> str | None:
        return self._active_adapters.get(base_key)

    def get_default_model(self) -> LoadedModelHandle | None:
        if self._default_model and self._default_model in self._models:
            return self._models[self._default_model]
        if self._models:
            return next(iter(self._models.values()))
        return None

    def get_model(self, key: str) -> LoadedModelHandle | None:
        return self._models.get(key)

    def load_base_model(
        self,
        model_key: str,
        gpu_memory_utilization: float = 0.9,
        set_default: bool = False,
    ) -> LoadedModelHandle:
        """Load a base model by its config key."""
        if model_key in self._models:
            logger.info("Model '%s' already loaded", model_key)
            return self._models[model_key]

        model_config = self.config.models.base_models.get(model_key)
        if model_config is None:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(self.config.models.base_models.keys())}"
            )

        self.memory.register(model_key, model_config.size_gb, "base")

        try:
            handle = load_model(
                model_key=model_key,
                config=model_config,
                model_dir=self.config.deployment.model_dir,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception:
            self.memory.unregister(model_key)
            raise

        self._models[model_key] = handle
        self._active_adapters[model_key] = None

        if set_default or self._default_model is None:
            self._default_model = model_key

        return handle

    def unload_model(self, model_key: str) -> None:
        """Unload a model and free its memory registration."""
        handle = self._models.pop(model_key, None)
        if handle is None:
            logger.warning("Model '%s' is not loaded", model_key)
            return

        self.memory.unregister(model_key)
        self._active_adapters.pop(model_key, None)

        if self._default_model == model_key:
            self._default_model = next(iter(self._models), None)

        logger.info("Model '%s' unloaded", model_key)

    def swap_adapter(self, base_key: str, adapter_key: str | None) -> None:
        """Hot-swap a LoRA adapter on a loaded base model."""
        if base_key not in self._models:
            raise ValueError(f"Base model '{base_key}' is not loaded")

        current = self._active_adapters.get(base_key)
        if current == adapter_key:
            return

        if adapter_key is not None:
            adapter_config = self.config.models.lora_adapters.get(adapter_key)
            if adapter_config is None:
                raise ValueError(
                    f"Unknown adapter '{adapter_key}'. "
                    f"Available: {list(self.config.models.lora_adapters.keys())}"
                )
            if adapter_config.base_model != base_key:
                raise ValueError(
                    f"Adapter '{adapter_key}' is trained on '{adapter_config.base_model}' "
                    f"and cannot be loaded on '{base_key}'. "
                    f"Cross-base adapter loading is not supported."
                )
            logger.info("Adapter '%s' set as active on '%s'", adapter_key, base_key)
        else:
            logger.info("Cleared adapter on '%s' (raw base model)", base_key)

        self._active_adapters[base_key] = adapter_key

    async def resolve_for_routing_async(self, decision: RoutingDecision) -> LoadedModelHandle:
        """Resolve a routing decision, loading specialists on demand if needed."""
        # Specialist routing
        if decision.specialist:
            spec_config = self._specialist_registry.get(decision.specialist)
            if spec_config is not None:
                try:
                    specialist = await self._specialist_loader.load(
                        decision.specialist, spec_config
                    )
                    logger.info("Specialist '%s' loaded for routing", decision.specialist)
                    # Wrap specialist in a LoadedModelHandle-compatible interface
                    # For now, fall back to default model since specialists need
                    # their own generate() path — the adapter handles I/O translation
                    # TODO: Build SpecialistModelHandle that wraps adapter + model
                except Exception as e:
                    logger.warning(
                        "Failed to load specialist '%s': %s, falling back to default",
                        decision.specialist, e,
                    )

            # Fall back to default model (specialist queries still get routed
            # through the default brain until specialist inference is fully wired)
            model = self.get_default_model()
            if model is None:
                raise RuntimeError("No models loaded")
            return model

        return self.resolve_for_routing(decision)

    def resolve_for_routing(self, decision: RoutingDecision) -> LoadedModelHandle:
        """Resolve a routing decision to a loaded model (sync version)."""
        # Specialist routing — sync fallback to default
        if decision.specialist:
            logger.info(
                "Specialist '%s' requested, falling back to default model (use async for on-demand loading)",
                decision.specialist,
            )
            model = self.get_default_model()
            if model is None:
                raise RuntimeError("No models loaded")
            return model

        # Brain routing
        if decision.base_model:
            model = self._models.get(decision.base_model)
            if model is not None:
                try:
                    self.swap_adapter(decision.base_model, decision.adapter)
                except ValueError as e:
                    logger.warning("Adapter swap failed: %s", e)
                return model
            else:
                logger.warning(
                    "Routed to '%s' but it's not loaded, falling back to default",
                    decision.base_model,
                )

        model = self.get_default_model()
        if model is None:
            raise RuntimeError("No models loaded")
        return model

    def get_model_for_domain(self, domain: str, is_hep: bool = False) -> LoadedModelHandle:
        """Resolve a domain to a loaded model (legacy Phase 1 interface)."""
        model = self.get_default_model()
        if model is None:
            raise RuntimeError("No models loaded")
        return model
