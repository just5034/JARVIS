"""Brain Manager — model loading, LoRA adapter swapping, memory tracking."""

from __future__ import annotations

import logging
import time

from jarvis.brains.memory_tracker import MemoryTracker
from jarvis.brains.model_loader import LoadedModelHandle, load_model
from jarvis.config import JarvisConfig

logger = logging.getLogger(__name__)


class BrainManager:
    """Manages base model loading and LoRA adapter hot-swapping."""

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self.memory = MemoryTracker(config.deployment.memory_budget)
        self._models: dict[str, LoadedModelHandle] = {}
        self._active_adapters: dict[str, str | None] = {}
        self._default_model: str | None = None

    @property
    def has_models(self) -> bool:
        return len(self._models) > 0

    def get_loaded_model_keys(self) -> list[str]:
        return list(self._models.keys())

    def get_default_model(self) -> LoadedModelHandle | None:
        if self._default_model and self._default_model in self._models:
            return self._models[self._default_model]
        # Return first loaded model as fallback
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

        # Check memory budget
        self.memory.register(model_key, model_config.size_gb, "base")

        try:
            handle = load_model(
                model_key=model_key,
                config=model_config,
                model_dir=self.config.deployment.model_dir,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception:
            # Roll back memory registration on failure
            self.memory.unregister(model_key)
            raise

        self._models[model_key] = handle

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

        if self._default_model == model_key:
            self._default_model = next(iter(self._models), None)

        logger.info("Model '%s' unloaded", model_key)

    def swap_adapter(self, base_key: str, adapter_key: str | None) -> None:
        """Hot-swap a LoRA adapter on a loaded base model."""
        # Phase 2: implement via vLLM's LoRA support
        raise NotImplementedError("Phase 2: LoRA adapter hot-swapping")

    def get_model_for_domain(self, domain: str, is_hep: bool = False) -> LoadedModelHandle:
        """Resolve a domain to a loaded model. Phase 1: returns default model."""
        # Phase 2 will use router config to map domain → base + adapter
        model = self.get_default_model()
        if model is None:
            raise RuntimeError("No models loaded")
        return model
