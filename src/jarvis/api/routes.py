"""Route handlers for the JARVIS API."""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from jarvis import __version__
from jarvis.api.models import (
    AdminLoadRequest,
    AdminLoadResponse,
    AdminMemoryResponse,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DeltaMessage,
    HealthResponse,
    JarvisMetadata,
    MemoryEntry,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    Usage,
)
from jarvis.brains.brain_manager import BrainManager
from jarvis.brains.model_loader import GenerationRequest
from jarvis.config import JarvisConfig

logger = logging.getLogger(__name__)
router = APIRouter()

_config: JarvisConfig | None = None
_brain_manager: BrainManager | None = None


def set_state(config: JarvisConfig, brain_manager: BrainManager) -> None:
    global _config, _brain_manager
    _config = config
    _brain_manager = brain_manager


def _get_config() -> JarvisConfig:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


def _get_brains() -> BrainManager:
    if _brain_manager is None:
        raise RuntimeError("BrainManager not initialized")
    return _brain_manager


@router.get("/health")
async def health() -> HealthResponse:
    config = _get_config()
    brains = _get_brains()
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=brains.get_loaded_model_keys(),
        memory_used_gb=brains.memory.used_gb,
        memory_available_gb=brains.memory.available_gb,
    )


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    config = _get_config()
    models = [ModelInfo(id="auto")]
    for name in config.router.domain_to_brain:
        models.append(ModelInfo(id=name))
    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    brains = _get_brains()
    start_time = time.monotonic()

    if not brains.has_models:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Load a model via POST /admin/load first.",
        )

    # Phase 1: use default model for all requests
    # Phase 2: will route based on request.model field
    model = brains.get_default_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No models available")

    # Build generation request
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    stop: list[str] | None = None
    if isinstance(request.stop, str):
        stop = [request.stop]
    elif isinstance(request.stop, list):
        stop = request.stop

    gen_request = GenerationRequest(
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 2048,
        stop=stop,
        n=request.n,
    )

    if request.stream:
        return _stream_response(model, gen_request, start_time)

    # Non-streaming: generate and return full response
    try:
        results = model.generate(gen_request)
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    elapsed_ms = (time.monotonic() - start_time) * 1000

    choices = []
    for i, result in enumerate(results):
        choices.append(
            Choice(
                index=i,
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        )

    # Use first result for usage stats
    first = results[0]
    usage = Usage(
        prompt_tokens=first.prompt_tokens,
        completion_tokens=first.completion_tokens,
        total_tokens=first.prompt_tokens + first.completion_tokens,
    )

    metadata = JarvisMetadata(
        inference_strategy="single_pass",
        num_candidates=request.n,
        base_model=model.model_key,
        latency_ms=round(elapsed_ms, 1),
    )

    return ChatCompletionResponse(
        model=model.model_key,
        choices=choices,
        usage=usage,
        jarvis_metadata=metadata,
    )


def _stream_response(model, gen_request: GenerationRequest, start_time: float):
    """Return an SSE streaming response."""

    async def event_generator():
        chunk_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # First chunk: role
        first_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model.model_key,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Generate — sync fallback, yields full text as one chunk
        # Phase 3 will use AsyncLLMEngine for true token-by-token streaming
        try:
            for result in model.generate_stream(gen_request):
                content_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model.model_key,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=result.text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"

                # Final chunk with finish_reason
                done_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model.model_key,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=result.finish_reason,
                        )
                    ],
                )
                yield f"data: {done_chunk.model_dump_json()}\n\n"
        except Exception as e:
            logger.error("Stream generation failed: %s", e)
            error_data = json.dumps({"error": {"message": str(e), "type": "server_error"}})
            yield f"data: {error_data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/admin/memory")
async def admin_memory() -> AdminMemoryResponse:
    config = _get_config()
    brains = _get_brains()
    models = [
        MemoryEntry(name=m["name"], size_gb=m["size_gb"], status=m["type"])
        for m in brains.memory.summary()
    ]
    return AdminMemoryResponse(
        total_gb=float(config.deployment.memory_budget.total_gb),
        used_gb=brains.memory.used_gb,
        available_gb=brains.memory.available_gb,
        models=models,
    )


@router.post("/admin/load")
async def admin_load(request: AdminLoadRequest) -> AdminLoadResponse:
    brains = _get_brains()

    if request.action == "load":
        try:
            start = time.monotonic()
            brains.load_base_model(request.model, set_default=True)
            elapsed_ms = (time.monotonic() - start) * 1000
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except MemoryError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ImportError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

        return AdminLoadResponse(
            model=request.model,
            action="load",
            status="loaded",
            memory_used_gb=brains.memory.used_gb,
        )

    elif request.action == "unload":
        brains.unload_model(request.model)
        return AdminLoadResponse(
            model=request.model,
            action="unload",
            status="unloaded",
            memory_used_gb=brains.memory.used_gb,
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{request.action}'. Use 'load' or 'unload'.",
        )
