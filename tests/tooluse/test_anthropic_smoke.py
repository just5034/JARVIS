"""End-to-end smoke test: Anthropic Messages API → shim → proxy → vLLM.

Sends Anthropic-shaped requests through the full stack and verifies
Anthropic-shaped responses come back. Skips when the shim isn't running.

Env:
    JARVIS_ANTHROPIC_SHIM_URL  default: http://localhost:8000
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

SHIM_URL = os.environ.get("JARVIS_ANTHROPIC_SHIM_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("JARVIS_TOOLUSE_MODEL", "qwen3.5-27b")


def _shim_alive() -> bool:
    try:
        r = httpx.get(f"{SHIM_URL}/anthropic/health", timeout=3.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _shim_alive(),
    reason=f"Anthropic shim not reachable at {SHIM_URL}",
)

TOOLS = [
    {
        "name": "list_dir",
        "description": "List the contents of a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute directory path"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a text file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "bash",
        "description": "Run a bash command and return stdout.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def _post(payload: dict) -> dict:
    r = httpx.post(f"{SHIM_URL}/v1/messages", json=payload, timeout=120.0)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
    return r.json()


def test_anthropic_text_response():
    """Simple text response in Anthropic format."""
    payload = {
        "model": MODEL,
        "system": "You are a helpful assistant. Answer briefly.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    data = _post(payload)

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["stop_reason"] == "end_turn"
    assert len(data["content"]) >= 1
    assert data["content"][0]["type"] == "text"
    assert "4" in data["content"][0]["text"]
    assert "usage" in data
    assert data["usage"]["input_tokens"] > 0


def test_anthropic_tool_call():
    """Tool call in Anthropic format — tool_use block with input."""
    payload = {
        "model": MODEL,
        "system": "You are a coding assistant. Use the provided tools.",
        "messages": [{"role": "user", "content": "List the contents of /tmp."}],
        "tools": TOOLS,
        "tool_choice": {"type": "auto"},
        "max_tokens": 512,
        "temperature": 0.2,
    }
    data = _post(payload)

    assert data["type"] == "message"
    assert data["stop_reason"] == "tool_use"

    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert tool_blocks, f"no tool_use blocks in: {json.dumps(data['content'])[:500]}"

    block = tool_blocks[0]
    assert block["name"] == "list_dir"
    assert "id" in block
    assert isinstance(block["input"], dict)
    assert "/tmp" in block["input"].get("path", "")


def test_anthropic_tool_result_continuation():
    """Full tool loop: call → result → continuation, all in Anthropic format."""
    # Step 1: get a tool call
    first = {
        "model": MODEL,
        "system": "You are a coding assistant. Use the provided tools.",
        "messages": [{"role": "user", "content": "What files are in /tmp? Use list_dir."}],
        "tools": TOOLS,
        "tool_choice": {"type": "auto"},
        "max_tokens": 512,
        "temperature": 0.2,
    }
    data1 = _post(first)
    assert data1["stop_reason"] == "tool_use"
    tool_blocks = [b for b in data1["content"] if b["type"] == "tool_use"]
    assert tool_blocks
    tool_block = tool_blocks[0]

    # Step 2: send tool result back, get continuation
    second = {
        "model": MODEL,
        "system": "You are a coding assistant. Use the provided tools.",
        "messages": [
            {"role": "user", "content": "What files are in /tmp? Use list_dir."},
            {"role": "assistant", "content": data1["content"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block["id"],
                        "content": "foo.txt\nbar.log\nproject/",
                    }
                ],
            },
        ],
        "tools": TOOLS,
        "tool_choice": {"type": "auto"},
        "max_tokens": 512,
        "temperature": 0.2,
    }
    data2 = _post(second)

    assert data2["type"] == "message"
    has_text = any(b["type"] == "text" for b in data2["content"])
    has_more_tools = any(b["type"] == "tool_use" for b in data2["content"])
    assert has_text or has_more_tools, f"empty continuation: {json.dumps(data2['content'])[:500]}"

    if has_text:
        text = " ".join(b["text"] for b in data2["content"] if b["type"] == "text").lower()
        assert any(name in text for name in ("foo", "bar", "project")), \
            f"continuation did not reference tool result: {text[:300]}"


def test_anthropic_streaming():
    """Streaming response returns proper Anthropic SSE events."""
    payload = {
        "model": MODEL,
        "system": "You are a helpful assistant. Answer briefly.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 128,
        "temperature": 0.1,
        "stream": True,
    }

    seen_events = set()
    text_content = ""

    with httpx.stream("POST", f"{SHIM_URL}/v1/messages", json=payload, timeout=120.0) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("event: "):
                seen_events.add(line[7:])
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_content += delta.get("text", "")
                except json.JSONDecodeError:
                    pass

    assert "message_start" in seen_events, f"missing message_start, got: {seen_events}"
    assert "content_block_start" in seen_events, f"missing content_block_start"
    assert "content_block_delta" in seen_events, f"missing content_block_delta"
    assert "content_block_stop" in seen_events, f"missing content_block_stop"
    assert "message_delta" in seen_events, f"missing message_delta"
    assert "message_stop" in seen_events, f"missing message_stop"
    assert len(text_content) > 0, "no text content received in stream"


def test_anthropic_ignores_extras():
    """Anthropic-specific fields (betas, thinking, cache_control) don't cause errors."""
    payload = {
        "model": MODEL,
        "system": [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 128,
        "temperature": 0.5,
        "thinking": {"type": "adaptive"},
        "metadata": {"user_id": "test-user"},
        "output_config": {"effort": "high"},
        "context_management": {"type": "auto"},
    }
    data = _post(payload)
    assert data["type"] == "message"
    assert len(data["content"]) >= 1
