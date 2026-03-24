"""On-demand specialist model loading with LRU eviction."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from jarvis.brains.memory_tracker import MemoryTracker
from jarvis.config import SpecialistConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedSpecialist:
    """A loaded specialist model with its adapter and metadata."""

    name: str
    config: SpecialistConfig
    model: Any  # The loaded model object (vLLM LLM, transformers model, etc.)
    adapter_type: str  # "text_llm", "protein_model", "dna_model"
    load_time: float


class SpecialistLoader:
    """Loads specialist models from SSD on demand, evicting LRU when needed."""

    def __init__(self, memory: MemoryTracker, model_dir: str = "/models") -> None:
        self.memory = memory
        self.model_dir = model_dir
        self._loaded: OrderedDict[str, LoadedSpecialist] = OrderedDict()

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def get(self, name: str) -> LoadedSpecialist | None:
        if name in self._loaded:
            # Move to end (most recently used)
            self._loaded.move_to_end(name)
            return self._loaded[name]
        return None

    def list_loaded(self) -> list[str]:
        return list(self._loaded.keys())

    async def load(self, name: str, config: SpecialistConfig) -> LoadedSpecialist:
        """Load a specialist model. Evicts LRU if memory is insufficient."""
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]

        # Evict LRU specialists until we have enough memory
        while not self.memory.can_load(config.size_gb) and self._loaded:
            evicted = await self.unload_lru()
            if evicted is None:
                break

        # Register memory
        try:
            self.memory.register(name, config.size_gb, "specialist")
        except MemoryError:
            raise MemoryError(
                f"Cannot load specialist '{name}' ({config.size_gb:.1f} GB): "
                f"insufficient memory even after eviction"
            )

        # Load the model
        start = time.monotonic()
        try:
            model = self._load_model(name, config)
        except Exception:
            self.memory.unregister(name)
            raise

        elapsed = time.monotonic() - start
        logger.info("Specialist '%s' loaded in %.1fs", name, elapsed)

        specialist = LoadedSpecialist(
            name=name,
            config=config,
            model=model,
            adapter_type=config.type,
            load_time=elapsed,
        )
        self._loaded[name] = specialist
        return specialist

    async def unload_lru(self) -> str | None:
        """Evict the least-recently-used specialist. Returns the evicted name."""
        if not self._loaded:
            return None

        # OrderedDict: first item is the LRU
        lru_name, lru_specialist = next(iter(self._loaded.items()))
        del self._loaded[lru_name]
        self.memory.unregister(lru_name)
        logger.info("Evicted specialist '%s' (LRU)", lru_name)
        return lru_name

    async def unload(self, name: str) -> None:
        """Explicitly unload a specialist."""
        if name in self._loaded:
            del self._loaded[name]
            self.memory.unregister(name)
            logger.info("Unloaded specialist '%s'", name)

    def _load_model(self, name: str, config: SpecialistConfig) -> Any:
        """Load the actual model. Dispatches based on specialist type."""
        if config.type == "text_llm":
            return self._load_text_llm(name, config)
        elif config.type in ("protein_model", "dna_model"):
            return self._load_transformers_model(name, config)
        else:
            raise ValueError(f"Unknown specialist type: {config.type}")

    def _load_text_llm(self, name: str, config: SpecialistConfig) -> Any:
        """Load a standard text LLM specialist via vLLM."""
        try:
            from vllm import LLM

            import os

            model_path = config.path
            if not os.path.exists(model_path):
                model_path = config.model_id

            quantization = config.quantization
            if quantization in ("nvfp4", "none", ""):
                quantization = None

            return LLM(
                model=model_path,
                max_model_len=8192,
                gpu_memory_utilization=0.3,  # Specialists get a small GPU slice
                trust_remote_code=True,
                quantization=quantization,
            )
        except ImportError:
            logger.warning("vLLM not available, returning placeholder for '%s'", name)
            return None

    def _load_transformers_model(self, name: str, config: SpecialistConfig) -> Any:
        """Load a non-text specialist (ESM3, Evo2) via transformers."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            import os

            model_path = config.path
            if not os.path.exists(model_path):
                model_path = config.model_id

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True
            ).to(device)

            return {"model": model, "tokenizer": tokenizer, "device": device}
        except ImportError:
            logger.warning("transformers not available, returning placeholder for '%s'", name)
            return None
        except Exception as e:
            logger.warning("Failed to load specialist '%s': %s", name, e)
            return None
