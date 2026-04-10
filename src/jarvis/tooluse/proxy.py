"""Thin pass-through proxy to a vLLM OpenAI server with tool-call support.

The proxy does NOT parse tool calls itself. vLLM does that when launched with
--enable-auto-tool-choice --tool-call-parser <parser>. This module just
forwards JSON, optionally streams the response back, and surfaces a few
configuration knobs via env vars so the experiment is self-contained.

Env vars:
    JARVIS_TOOLUSE_VLLM_URL   default: http://localhost:8290
    JARVIS_TOOLUSE_TIMEOUT_S  default: 300
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

VLLM_URL = os.environ.get("JARVIS_TOOLUSE_VLLM_URL", "http://localhost:8290").rstrip("/")
TIMEOUT_S = float(os.environ.get("JARVIS_TOOLUSE_TIMEOUT_S", "300"))

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Report proxy health and whether the upstream vLLM server is reachable."""
    upstream = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLLM_URL}/health")
            upstream = "ok" if r.status_code == 200 else f"status_{r.status_code}"
    except Exception as e:  # noqa: BLE001
        upstream = f"unreachable: {type(e).__name__}"
    return {"proxy": "ok", "upstream": upstream, "vllm_url": VLLM_URL}


@router.get("/v1/models")
async def list_models() -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{VLLM_URL}/v1/models")
            return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {e}")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward an OpenAI chat-completions request (with tools) to vLLM.

    We accept the raw JSON body rather than re-parsing through the schemas so
    that any field vLLM understands but our pydantic models don't yet (e.g.
    response_format, logprobs) still passes through. The schemas in
    jarvis.tooluse.schemas exist for clients/tests, not as a gate.
    """
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}")

    stream = bool(body.get("stream", False))
    url = f"{VLLM_URL}/v1/chat/completions"

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                r = await client.post(url, json=body)
                return JSONResponse(status_code=r.status_code, content=r.json())
        except httpx.HTTPError as e:
            logger.error("Upstream call failed: %s", e)
            raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    # For streaming, use a long read timeout so large generations aren't killed,
    # but keep a short connect timeout so we fail fast if vLLM is down.
    stream_timeout = httpx.Timeout(connect=10.0, read=TIMEOUT_S, write=10.0, pool=10.0)

    async def event_stream():
        try:
            async with httpx.AsyncClient(timeout=stream_timeout) as client:
                async with client.stream("POST", url, json=body) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", errors="replace")
                        yield f"data: {{\"error\": {text!r}}}\n\n".encode()
                        return
                    async for chunk in r.aiter_raw():
                        if chunk:
                            yield chunk
        except httpx.HTTPError as e:
            logger.error("Upstream stream failed: %s", e)
            yield f"data: {{\"error\": \"{e}\"}}\n\n".encode()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
