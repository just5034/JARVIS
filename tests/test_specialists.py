"""Tests for Phase 5: Specialist ecosystem."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.brains.memory_tracker import MemoryTracker
from jarvis.config import JarvisConfig, MemoryBudgetConfig, SpecialistConfig, load_config
from jarvis.specialists.adapters.esm3 import ESM3Adapter
from jarvis.specialists.adapters.evo2 import Evo2Adapter
from jarvis.specialists.adapters.text_llm import TextLLMAdapter
from jarvis.specialists.loader import SpecialistLoader
from jarvis.specialists.registry import SpecialistRegistry


@pytest.fixture
def config() -> JarvisConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    return load_config(config_dir)


# --- SpecialistRegistry ---


class TestSpecialistRegistry:
    def test_list_available(self, config: JarvisConfig) -> None:
        registry = SpecialistRegistry(config.models)
        available = registry.list_available()
        assert "chemistry" in available
        assert "biology" in available
        assert "protein" in available
        assert "genomics" in available

    def test_get_specialist(self, config: JarvisConfig) -> None:
        registry = SpecialistRegistry(config.models)
        chem = registry.get("chemistry")
        assert chem is not None
        assert chem.model_id == "AI4Chem/ChemLLM-7B-Chat"
        assert chem.type == "text_llm"

    def test_get_nonexistent(self, config: JarvisConfig) -> None:
        registry = SpecialistRegistry(config.models)
        assert registry.get("nonexistent") is None

    def test_requires_adapter(self, config: JarvisConfig) -> None:
        registry = SpecialistRegistry(config.models)
        assert registry.requires_adapter("protein") is True
        assert registry.requires_adapter("chemistry") is False


# --- SpecialistLoader ---


class TestSpecialistLoader:
    def test_initial_state(self) -> None:
        budget = MemoryBudgetConfig(total_gb=40, reserved_os_gb=2, reserved_framework_gb=4, safety_margin_gb=2)
        memory = MemoryTracker(budget)
        loader = SpecialistLoader(memory)
        assert not loader.is_loaded("chemistry")
        assert loader.list_loaded() == []

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        budget = MemoryBudgetConfig(total_gb=20, reserved_os_gb=2, reserved_framework_gb=2, safety_margin_gb=1)
        memory = MemoryTracker(budget)
        loader = SpecialistLoader(memory)

        # Manually load two "specialists" bypassing actual model loading
        from jarvis.specialists.loader import LoadedSpecialist
        loader._loaded["a"] = LoadedSpecialist(name="a", config=None, model=None, adapter_type="text_llm", load_time=0.1)
        memory.register("a", 5.0, "specialist")
        loader._loaded["b"] = LoadedSpecialist(name="b", config=None, model=None, adapter_type="text_llm", load_time=0.1)
        memory.register("b", 5.0, "specialist")

        assert loader.is_loaded("a")
        assert loader.is_loaded("b")

        # Evict LRU — should be "a" (first in)
        evicted = await loader.unload_lru()
        assert evicted == "a"
        assert not loader.is_loaded("a")
        assert loader.is_loaded("b")
        assert memory.used_gb == 5.0

    @pytest.mark.asyncio
    async def test_unload_explicit(self) -> None:
        budget = MemoryBudgetConfig(total_gb=40, reserved_os_gb=2, reserved_framework_gb=2, safety_margin_gb=1)
        memory = MemoryTracker(budget)
        loader = SpecialistLoader(memory)

        from jarvis.specialists.loader import LoadedSpecialist
        loader._loaded["test"] = LoadedSpecialist(name="test", config=None, model=None, adapter_type="text_llm", load_time=0.1)
        memory.register("test", 3.5, "specialist")

        await loader.unload("test")
        assert not loader.is_loaded("test")
        assert memory.used_gb == 0.0


# --- ESM3 Adapter ---


class TestESM3Adapter:
    def test_parse_sequence(self) -> None:
        adapter = ESM3Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": "Predict the structure of protein sequence MVLSPADKTNVKAAWGKVGA"}
        ])
        assert result["sequence"] == "MVLSPADKTNVKAAWGKVGA"
        assert result["task"] == "structure_prediction"

    def test_parse_fasta(self) -> None:
        adapter = ESM3Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": ">myprotein\nMVLSPADKTNVKAAWGKVGA"}
        ])
        assert result["sequence"] == "MVLSPADKTNVKAAWGKVGA"

    def test_parse_no_sequence(self) -> None:
        adapter = ESM3Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": "What is a protein?"}
        ])
        assert result["sequence"] is None

    def test_task_detection(self) -> None:
        adapter = ESM3Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": "What is the function of MVLSPADKTNVKAAWGKVGA"}
        ])
        assert result["task"] == "function_prediction"

    def test_format_output(self) -> None:
        adapter = ESM3Adapter()
        output = adapter.format_output({
            "sequence": "MVLSPAD",
            "confidence": 0.85,
        })
        assert "MVLSPAD" in output
        assert "0.85" in output

    def test_format_none(self) -> None:
        adapter = ESM3Adapter()
        output = adapter.format_output(None)
        assert "Unable to process" in output


# --- Evo2 Adapter ---


class TestEvo2Adapter:
    def test_parse_dna_sequence(self) -> None:
        adapter = Evo2Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": "Analyze this DNA: ACGTACGTACGTACGTACGTACGT"}
        ])
        assert result["sequence"] == "ACGTACGTACGTACGTACGTACGT"

    def test_parse_mutation(self) -> None:
        adapter = Evo2Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": "What is the effect of mutation G>A at position 12345 in BRCA1?"}
        ])
        assert len(result["mutations"]) == 1
        assert result["mutations"][0]["ref"] == "G"
        assert result["mutations"][0]["alt"] == "A"
        assert result["mutations"][0]["position"] == 12345
        assert result["task"] == "variant_effect"

    def test_parse_no_sequence(self) -> None:
        adapter = Evo2Adapter()
        result = adapter.parse_input([
            {"role": "user", "content": "What is DNA?"}
        ])
        assert result["sequence"] is None

    def test_format_variant_effects(self) -> None:
        adapter = Evo2Adapter()
        output = adapter.format_output({
            "sequence": "ACGT" * 10,
            "variant_effects": [
                {"ref": "G", "alt": "A", "position": 100, "score": 0.95, "label": "pathogenic"}
            ],
        })
        assert "pathogenic" in output
        assert "G100A" in output

    def test_format_none(self) -> None:
        adapter = Evo2Adapter()
        output = adapter.format_output(None)
        assert "Unable to process" in output


# --- TextLLM Adapter ---


class TestTextLLMAdapter:
    def test_passthrough(self) -> None:
        adapter = TextLLMAdapter()
        messages = [{"role": "user", "content": "Hello"}]
        assert adapter.parse_input(messages) == messages

    def test_format_output(self) -> None:
        adapter = TextLLMAdapter()
        assert adapter.format_output("Hello world") == "Hello world"
