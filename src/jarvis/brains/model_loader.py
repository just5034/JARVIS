"""Model loader — loads models via vLLM for inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.config import BaseModelConfig

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)

# Lazy vLLM import — not available on Windows or without [serving] extras
_vllm_available: bool | None = None


def _check_vllm() -> bool:
    global _vllm_available
    if _vllm_available is None:
        try:
            import vllm  # noqa: F401

            _vllm_available = True
        except ImportError:
            _vllm_available = False
    return _vllm_available


@dataclass
class GenerationRequest:
    """Parameters for a single generation call."""

    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    stop: list[str] | None = None
    n: int = 1


@dataclass
class GenerationResult:
    """Result from a single generation call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class LoadedModelHandle:
    """Wrapper around a loaded vLLM model with generation methods."""

    def __init__(
        self,
        model_key: str,
        config: BaseModelConfig,
        model_dir: str,
        llm: Any = None,
    ) -> None:
        self.model_key = model_key
        self.config = config
        self.model_dir = model_dir
        self._llm = llm

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def _build_sampling_params(self, request: GenerationRequest) -> "SamplingParams":
        from vllm import SamplingParams

        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or [],
            n=request.n,
        )

    def _format_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        """Simple chat template fallback. vLLM's tokenizer.apply_chat_template is preferred."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        if self._llm is None:
            raise RuntimeError(f"Model '{self.model_key}' is not loaded")

        sampling_params = self._build_sampling_params(request)

        # Prefer chat messages → tokenizer chat template
        if request.messages:
            try:
                prompt = self._llm.get_tokenizer().apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = self._format_chat_prompt(request.messages)
        else:
            prompt = request.prompt or ""

        outputs: list[RequestOutput] = self._llm.generate([prompt], sampling_params)
        output = outputs[0]

        results = []
        for completion in output.outputs:
            results.append(
                GenerationResult(
                    text=completion.text,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    finish_reason=completion.finish_reason or "stop",
                )
            )
        return results

    def generate_stream(self, request: GenerationRequest):
        """Yields text chunks as they are generated."""
        if self._llm is None:
            raise RuntimeError(f"Model '{self.model_key}' is not loaded")

        # vLLM's LLM class doesn't natively support streaming.
        # For streaming, we need the AsyncLLMEngine — see AsyncModelLoader.
        # This sync fallback yields the full result as a single chunk.
        results = self.generate(request)
        if results:
            yield results[0]


def load_model(
    model_key: str,
    config: BaseModelConfig,
    model_dir: str,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
) -> LoadedModelHandle:
    """Load a model via vLLM and return a handle for generation."""
    if not _check_vllm():
        raise ImportError(
            "vLLM is not installed. Install with: pip install 'jarvis[serving]'"
        )

    from vllm import LLM

    import os

    model_path = config.path
    if not os.path.isabs(model_path):
        model_path = os.path.join(model_dir, model_path)

    # If local path doesn't exist, fall back to HuggingFace model_id
    if not os.path.exists(model_path):
        logger.info(
            "Local path %s not found, loading from HuggingFace: %s",
            model_path,
            config.model_id,
        )
        model_path = config.model_id

    logger.info("Loading model '%s' from %s", model_key, model_path)
    start = time.monotonic()

    quantization = config.quantization if config.quantization != "nvfp4" else None
    # nvfp4 requires Blackwell hardware — on non-Blackwell, load without quantization
    # and let vLLM handle dtype automatically

    llm = LLM(
        model=model_path,
        max_model_len=config.recommended_max_context,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        quantization=quantization,
        trust_remote_code=True,
    )

    elapsed = time.monotonic() - start
    logger.info("Model '%s' loaded in %.1fs", model_key, elapsed)

    return LoadedModelHandle(
        model_key=model_key,
        config=config,
        model_dir=model_dir,
        llm=llm,
    )
