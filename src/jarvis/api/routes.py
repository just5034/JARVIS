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
from jarvis.router.router import Router

logger = logging.getLogger(__name__)
router = APIRouter()

_config: JarvisConfig | None = None
_brain_manager: BrainManager | None = None
_router: Router | None = None


def set_state(
    config: JarvisConfig,
    brain_manager: BrainManager,
    query_router: Router | None = None,
) -> None:
    global _config, _brain_manager, _router
    _config = config
    _brain_manager = brain_manager
    _router = query_router


def _get_config() -> JarvisConfig:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


def _get_brains() -> BrainManager:
    if _brain_manager is None:
        raise RuntimeError("BrainManager not initialized")
    return _brain_manager


def _get_router() -> Router | None:
    return _router


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
    query_router = _get_router()
    start_time = time.monotonic()

    if not brains.has_models:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Load a model via POST /admin/load first.",
        )

    # Extract query text for routing
    user_messages = [m for m in request.messages if m.role == "user"]
    system_messages = [m for m in request.messages if m.role == "system"]
    query_text = user_messages[-1].content if user_messages else ""
    system_prompt = system_messages[0].content if system_messages else None

    # Route the query
    routing_decision = None
    if query_router is not None:
        force_domain = None
        if request.model and request.model not in ("", "auto"):
            force_domain = request.model

        routing_decision = query_router.route(
            query=query_text,
            system_prompt=system_prompt,
            force_domain=force_domain,
        )
        model = brains.resolve_for_routing(routing_decision)
    else:
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
        return _stream_response(model, gen_request, start_time, routing_decision)

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

    first = results[0]
    usage = Usage(
        prompt_tokens=first.prompt_tokens,
        completion_tokens=first.completion_tokens,
        total_tokens=first.prompt_tokens + first.completion_tokens,
    )

    metadata = JarvisMetadata(
        routed_domain=routing_decision.domain if routing_decision else None,
        difficulty=routing_decision.difficulty if routing_decision else None,
        inference_strategy="single_pass",
        num_candidates=request.n,
        base_model=model.model_key,
        adapter=routing_decision.adapter if routing_decision else None,
        latency_ms=round(elapsed_ms, 1),
    )

    return ChatCompletionResponse(
        model=model.model_key,
        choices=choices,
        usage=usage,
        jarvis_metadata=metadata,
    )


def _stream_response(model, gen_request, start_time, routing_decision=None):
    """Return an SSE streaming response."""

    async def event_generator():
        chunk_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

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
