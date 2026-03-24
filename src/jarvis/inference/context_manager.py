"""KV cache management — configuration computation for difficulty-aware inference."""

from __future__ import annotations

from jarvis.config import InferenceConfig


class ContextManager:
    """Computes KV cache configuration based on difficulty and sampling strategy."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def get_kv_config(self, difficulty: str, num_candidates: int) -> dict:
        """Compute recommended KV cache settings for a given inference strategy.

        Returns a dict with keys that map to vLLM engine configuration.
        """
        level = self.config.difficulty_levels.get(difficulty)
        if level is None:
            level = self.config.difficulty_levels.get("easy")

        kv_dtype = level.kv_cache_dtype if level else "auto"
        max_context = level.max_context_length if level else 32768

        # Use 2-bit quantization for hard queries if configured
        kv_quant_bits = None
        if level and level.kv_quant_bits:
            kv_quant_bits = level.kv_quant_bits

        return {
            "kv_cache_dtype": kv_dtype,
            "max_model_len": max_context,
            "kv_quant_bits": kv_quant_bits,
            "enable_prefix_caching": num_candidates > 1,
            "ssd_offload_enabled": self.config.context_management.ssd_offload_enabled,
            "ssd_offload_path": self.config.context_management.ssd_offload_path,
        }

    def estimate_kv_memory_gb(
        self,
        context_length: int,
        num_candidates: int,
        kv_dtype: str = "fp8",
    ) -> float:
        """Estimate KV cache memory usage in GB.

        Based on a 32B model with 8 KV heads, 128 dim/head, 64 layers.
        Adjust the base cost per token based on dtype.
        """
        # Bytes per token per layer for one KV head pair (key + value)
        # 2 (K+V) * num_heads * head_dim * dtype_bytes
        # For 32B: 2 * 8 * 128 = 2048 bytes per layer at FP16
        bytes_per_token_per_layer = {
            "fp16": 2048,
            "fp8": 1024,
            "auto": 1024,  # Default to FP8
        }

        base_bytes = bytes_per_token_per_layer.get(kv_dtype, 1024)
        num_layers = 64  # 32B model

        # 2-bit KV is ~4x smaller than FP8
        if kv_dtype in ("fp8", "auto") and num_candidates > 8:
            # For large N, assume 2-bit KV might be used
            base_bytes = base_bytes // 4

        total_bytes = base_bytes * num_layers * context_length * num_candidates
        return total_bytes / (1024 ** 3)
