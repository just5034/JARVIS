"""OpenAI-compatible request/response schemas for the JARVIS API."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# --- Requests ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1


# --- Responses ---


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class JarvisMetadata(BaseModel):
    routed_domain: str | None = None
    difficulty: str | None = None
    inference_strategy: str | None = None
    num_candidates: int | None = None
    verification_score: float | None = None
    base_model: str | None = None
    adapter: str | None = None
    latency_ms: float | None = None


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: Usage
    jarvis_metadata: JarvisMetadata | None = None


# --- Streaming ---


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice]


# --- Other endpoints ---


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "jarvis"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    memory_used_gb: float
    memory_available_gb: float


class MemoryEntry(BaseModel):
    name: str
    size_gb: float
    status: str


class AdminMemoryResponse(BaseModel):
    total_gb: float
    used_gb: float
    available_gb: float
    models: list[MemoryEntry]


class AdminLoadRequest(BaseModel):
    model: str
    action: str = "load"


class AdminLoadResponse(BaseModel):
    model: str
    action: str
    status: str
    memory_used_gb: float


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
