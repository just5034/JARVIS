"""OpenAI-compatible chat schemas with tool-calling fields.

Kept separate from jarvis.api.models so the experiment cannot break the
existing /v1/chat/completions schema used by ongoing evals.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# --- Tool definitions (request side) ---


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


# --- Tool calls (response side) ---


class ToolCallFunction(BaseModel):
    name: str
    # OpenAI convention: arguments is a JSON-encoded *string*, not a dict.
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


# --- Messages ---


class ToolMessage(BaseModel):
    """A chat message that can carry assistant tool_calls or a tool result.

    role: "system" | "user" | "assistant" | "tool"
    - assistant messages may include tool_calls
    - tool messages must include tool_call_id and a content string (the result)
    """

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


# --- Request ---


class ToolChatRequest(BaseModel):
    model: str = "qwen3.5-27b"
    messages: list[ToolMessage]
    tools: list[Tool] | None = None
    # "auto" | "none" | "required" | {"type":"function","function":{"name":...}}
    tool_choice: str | dict[str, Any] | None = "auto"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = 2048
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1


# --- Response ---


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int
    message: ToolMessage
    finish_reason: str | None = None


class ToolChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
