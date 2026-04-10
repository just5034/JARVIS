"""Translate between Anthropic Messages API and OpenAI Chat Completions API.

This module converts:
  - Anthropic requests → OpenAI requests (to send to vLLM)
  - OpenAI responses → Anthropic responses (to return to CodeAgent)

The translation is mechanical — same information, different JSON shapes.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


# ─── Request translation: Anthropic → OpenAI ───


def translate_request(anthropic_body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format."""
    messages: list[dict[str, Any]] = []

    # System prompt: Anthropic has it as a separate field (string or array of blocks)
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Array of text blocks: [{"type": "text", "text": "...", ...}, ...]
            text_parts = []
            for block in system:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            messages.append({"role": "system", "content": "\n".join(text_parts)})

    # Messages: convert content blocks to OpenAI format
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            converted = _convert_content_blocks(role, content, messages)
            if converted:
                messages.extend(converted)
        else:
            messages.append({"role": role, "content": content or ""})

    # Tools: Anthropic uses input_schema, OpenAI uses parameters under function
    tools = None
    if anthropic_body.get("tools"):
        tools = []
        for tool in anthropic_body["tools"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

    # Tool choice
    tool_choice = None
    tc = anthropic_body.get("tool_choice")
    if tc:
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "auto":
                tool_choice = "auto"
            elif tc_type == "any":
                tool_choice = "required"
            elif tc_type == "tool":
                tool_choice = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
            elif tc_type == "none":
                tool_choice = "none"
        elif isinstance(tc, str):
            tool_choice = tc

    result: dict[str, Any] = {
        "model": anthropic_body.get("model", "qwen3.5-27b"),
        "messages": messages,
        "max_tokens": anthropic_body.get("max_tokens", 4096),
        "stream": anthropic_body.get("stream", False),
    }

    if tools:
        result["tools"] = tools
    if tool_choice is not None:
        result["tool_choice"] = tool_choice

    temp = anthropic_body.get("temperature")
    if temp is not None:
        result["temperature"] = temp

    top_p = anthropic_body.get("top_p")
    if top_p is not None:
        result["top_p"] = top_p

    stop = anthropic_body.get("stop_sequences")
    if stop:
        result["stop"] = stop

    return result


def _convert_content_blocks(
    role: str,
    blocks: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic content block arrays to OpenAI message(s).

    An Anthropic assistant message can contain mixed text + tool_use blocks.
    An Anthropic user message can contain tool_result blocks.
    """
    result: list[dict[str, Any]] = []

    if role == "assistant":
        text_parts = []
        tool_calls = []
        for block in blocks:
            btype = block.get("type", "text")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif btype == "thinking":
                # Thinking blocks — ignore for OpenAI, or prepend to text
                pass

        msg: dict[str, Any] = {"role": "assistant"}
        msg["content"] = "\n".join(text_parts) if text_parts else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        result.append(msg)

    elif role == "user":
        text_parts = []
        tool_results = []
        for block in blocks:
            btype = block.get("type", "text")
            if btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    # Content can be array of blocks
                    content = "\n".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(content),
                })
            elif btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image":
                # Images — skip, vLLM text model can't handle them
                text_parts.append("[image omitted]")

        # Emit any text as a user message first
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})
        # Emit tool results as separate tool messages (OpenAI format)
        result.extend(tool_results)

    return result


# ─── Response translation: OpenAI → Anthropic ───


def translate_response(openai_resp: dict[str, Any], model: str = "") -> dict[str, Any]:
    """Convert an OpenAI chat completion response to Anthropic Messages format."""
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = openai_resp.get("usage", {})

    content: list[dict[str, Any]] = []

    # Text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls → tool_use blocks
    for tc in message.get("tool_calls", []) or []:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {"raw": func.get("arguments", "")}

        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": func.get("name", ""),
            "input": input_data,
        })

    # Determine stop reason
    finish = choice.get("finish_reason", "end_turn")
    if finish == "tool_calls" or message.get("tool_calls"):
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    elif finish == "stop":
        stop_reason = "end_turn"
    else:
        stop_reason = "end_turn"

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model or openai_resp.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ─── Streaming translation: OpenAI SSE → Anthropic SSE ───


class StreamTranslator:
    """Stateful translator that converts OpenAI streaming chunks to Anthropic SSE events.

    Feed it OpenAI SSE lines via feed_line(), it yields Anthropic SSE lines.
    """

    def __init__(self, model: str = ""):
        self.model = model
        self.block_index = 0
        self.current_block_type: str | None = None  # "text" or "tool_use"
        self.started = False
        self.tool_call_ids: dict[int, str] = {}  # OpenAI tool index → id
        self.tool_call_names: dict[int, str] = {}
        self.tool_call_started: set[int] = set()
        self._input_tokens = 0
        self._output_tokens = 0

    def start_event(self) -> list[str]:
        """Emit the message_start event."""
        self.started = True
        event = {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }
        return [f"event: message_start\ndata: {json.dumps(event)}\n\n"]

    def feed_chunk(self, openai_chunk: dict[str, Any]) -> list[str]:
        """Process one OpenAI streaming chunk, return zero or more Anthropic SSE events."""
        events: list[str] = []

        if not self.started:
            events.extend(self.start_event())

        # Track usage
        usage = openai_chunk.get("usage")
        if usage:
            self._input_tokens = usage.get("prompt_tokens", self._input_tokens)
            self._output_tokens = usage.get("completion_tokens", self._output_tokens)

        choices = openai_chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Text content delta
        text = delta.get("content")
        if text:
            if self.current_block_type != "text":
                # Close previous block if needed
                if self.current_block_type is not None:
                    events.append(self._block_stop())
                events.append(self._block_start_text())
            events.append(self._text_delta(text))

        # Tool call deltas
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                tc_func = tc.get("function", {})
                tc_id = tc.get("id")

                if tc_id:
                    self.tool_call_ids[tc_index] = tc_id
                if tc_func.get("name"):
                    self.tool_call_names[tc_index] = tc_func["name"]

                # Start a new tool_use block if we haven't for this tool index
                if tc_index not in self.tool_call_started:
                    if self.current_block_type is not None:
                        events.append(self._block_stop())
                    events.append(self._block_start_tool_use(tc_index))
                    self.tool_call_started.add(tc_index)

                # Emit argument fragments as input_json_delta
                args_frag = tc_func.get("arguments", "")
                if args_frag:
                    events.append(self._input_json_delta(args_frag))

        # Finish
        if finish_reason:
            # Close current block
            if self.current_block_type is not None:
                events.append(self._block_stop())

            # Map finish reason
            if finish_reason == "tool_calls" or self.tool_call_started:
                stop_reason = "tool_use"
            elif finish_reason == "length":
                stop_reason = "max_tokens"
            else:
                stop_reason = "end_turn"

            events.append(self._message_delta(stop_reason))
            events.append(self._message_stop())

        return events

    def _block_start_text(self) -> str:
        idx = self.block_index
        self.current_block_type = "text"
        event = {
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""},
        }
        self.block_index += 1
        return f"event: content_block_start\ndata: {json.dumps(event)}\n\n"

    def _block_start_tool_use(self, tc_index: int) -> str:
        idx = self.block_index
        self.current_block_type = "tool_use"
        event = {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "tool_use",
                "id": self.tool_call_ids.get(tc_index, f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": self.tool_call_names.get(tc_index, ""),
                "input": {},
            },
        }
        self.block_index += 1
        return f"event: content_block_start\ndata: {json.dumps(event)}\n\n"

    def _text_delta(self, text: str) -> str:
        event = {
            "type": "content_block_delta",
            "index": self.block_index - 1,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"

    def _input_json_delta(self, partial_json: str) -> str:
        event = {
            "type": "content_block_delta",
            "index": self.block_index - 1,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        }
        return f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"

    def _block_stop(self) -> str:
        event = {
            "type": "content_block_stop",
            "index": self.block_index - 1,
        }
        self.current_block_type = None
        return f"event: content_block_stop\ndata: {json.dumps(event)}\n\n"

    def _message_delta(self, stop_reason: str) -> str:
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self._output_tokens},
        }
        return f"event: message_delta\ndata: {json.dumps(event)}\n\n"

    def _message_stop(self) -> str:
        event = {"type": "message_stop"}
        return f"event: message_stop\ndata: {json.dumps(event)}\n\n"
