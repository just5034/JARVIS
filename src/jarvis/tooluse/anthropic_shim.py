"""Anthropic Messages API shim for CodeAgent compatibility.

Exposes POST /v1/messages that CodeAgent can talk to directly via:
    ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=dummy

Translates Anthropic → OpenAI, forwards to the vLLM proxy, then translates
the OpenAI response back to Anthropic format. Supports both streaming and
non-streaming.

Gracefully ignores Anthropic-specific fields that don't apply to local models:
betas, cache_control, thinking, context_management, metadata, output_config.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from jarvis.tooluse.anthropic_translate import (
    StreamTranslator,
    translate_request,
    translate_response,
)

logger = logging.getLogger(__name__)

# The upstream is the OpenAI-compatible proxy (which itself forwards to vLLM).
OPENAI_PROXY_URL = os.environ.get(
    "JARVIS_ANTHROPIC_SHIM_UPSTREAM", "http://localhost:8001"
).rstrip("/")
TIMEOUT_S = float(os.environ.get("JARVIS_TOOLUSE_TIMEOUT_S", "300"))

router = APIRouter()


@router.get("/anthropic/health")
async def anthropic_health() -> dict[str, str]:
    upstream = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OPENAI_PROXY_URL}/health")
            upstream = "ok" if r.status_code == 200 else f"status_{r.status_code}"
    except Exception as e:  # noqa: BLE001
        upstream = f"unreachable: {type(e).__name__}"
    return {"shim": "ok", "upstream": upstream}


@router.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API endpoint.

    Accepts the same request shape as https://docs.anthropic.com/en/api/messages
    and returns responses in the same format. CodeAgent talks to this endpoint.
    """
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}")

    # Extract model name before translation (for response metadata)
    model = body.get("model", "qwen3.5-27b")
    stream = body.get("stream", False)

    # Translate Anthropic request → OpenAI request
    try:
        openai_body = translate_request(body)
    except Exception as e:  # noqa: BLE001
        logger.error("Request translation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"request translation error: {e}")

    if not stream:
        return await _non_streaming(openai_body, model)
    else:
        return _streaming(openai_body, model)


async def _non_streaming(openai_body: dict[str, Any], model: str) -> JSONResponse:
    """Forward as non-streaming, translate response back."""
    openai_body["stream"] = False
    url = f"{OPENAI_PROXY_URL}/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(url, json=openai_body)
    except httpx.HTTPError as e:
        logger.error("Upstream call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    if r.status_code != 200:
        # Forward error as Anthropic-shaped error
        return JSONResponse(
            status_code=r.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": r.text[:1000],
                },
            },
        )

    openai_resp = r.json()
    anthropic_resp = translate_response(openai_resp, model=model)
    return JSONResponse(content=anthropic_resp)


def _streaming(openai_body: dict[str, Any], model: str) -> StreamingResponse:
    """Forward as streaming, translate SSE events back to Anthropic format."""
    openai_body["stream"] = True
    url = f"{OPENAI_PROXY_URL}/v1/chat/completions"

    stream_timeout = httpx.Timeout(connect=10.0, read=TIMEOUT_S, write=10.0, pool=10.0)

    async def event_stream():
        translator = StreamTranslator(model=model)

        try:
            async with httpx.AsyncClient(timeout=stream_timeout) as client:
                async with client.stream("POST", url, json=openai_body) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", errors="replace")
                        error_event = {
                            "type": "error",
                            "error": {"type": "api_error", "message": text[:1000]},
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                        return

                    buffer = ""
                    async for raw_chunk in r.aiter_text():
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                # Ensure we've sent final events
                                if translator.started and translator.current_block_type is not None:
                                    yield translator._block_stop()
                                    yield translator._message_delta("end_turn")
                                    yield translator._message_stop()
                                return
                            if not line.startswith("data: "):
                                continue

                            json_str = line[6:]  # Strip "data: " prefix
                            try:
                                chunk = json.loads(json_str)
                            except json.JSONDecodeError:
                                continue

                            for event_str in translator.feed_chunk(chunk):
                                yield event_str

        except httpx.HTTPError as e:
            logger.error("Upstream stream failed: %s", e)
            error_event = {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
