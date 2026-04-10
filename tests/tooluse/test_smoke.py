"""Tool-use smoke test against a running JARVIS tool-use proxy.

Assumes the proxy is already running (the SLURM script launches it before
invoking pytest). Skips itself cleanly when the proxy is not reachable so it
will not fire during a normal local test run.

Env:
    JARVIS_TOOLUSE_PROXY_URL  default: http://localhost:8001
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

PROXY_URL = os.environ.get("JARVIS_TOOLUSE_PROXY_URL", "http://localhost:8001").rstrip("/")
MODEL = os.environ.get("JARVIS_TOOLUSE_MODEL", "qwen3.5-27b")


def _proxy_alive() -> bool:
    try:
        r = httpx.get(f"{PROXY_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _proxy_alive(),
    reason=f"tool-use proxy not reachable at {PROXY_URL}",
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def _post(payload: dict) -> dict:
    r = httpx.post(f"{PROXY_URL}/v1/chat/completions", json=payload, timeout=120.0)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def test_single_tool_call():
    """Model should emit a structured tool_call for an obviously-tool-shaped task."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Use the provided tools."},
            {"role": "user", "content": "List the contents of /tmp."},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    data = _post(payload)
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")
    assert tool_calls, f"expected tool_calls, got: {json.dumps(msg)[:500]}"
    call = tool_calls[0]
    assert call["function"]["name"] == "list_dir", f"wrong tool: {call['function']['name']}"
    args = json.loads(call["function"]["arguments"])  # must be valid JSON
    assert "path" in args and "/tmp" in args["path"]


def test_tool_result_continuation():
    """Model should consume a tool_result and continue coherently."""
    first = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Use the provided tools."},
            {"role": "user", "content": "What files are in /tmp? Use list_dir."},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    data = _post(first)
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")
    assert tool_calls, "expected first turn to be a tool call"
    call = tool_calls[0]

    second = {
        "model": MODEL,
        "messages": [
            first["messages"][0],
            first["messages"][1],
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": "foo.txt\nbar.log\nproject/",
            },
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    data2 = _post(second)
    msg2 = data2["choices"][0]["message"]
    # Either: model returns text summarizing the listing, OR makes another call.
    has_text = bool(msg2.get("content"))
    has_more_calls = bool(msg2.get("tool_calls"))
    assert has_text or has_more_calls, f"empty follow-up: {json.dumps(msg2)[:500]}"
    if has_text:
        body = msg2["content"].lower()
        assert any(name in body for name in ("foo", "bar", "project")), \
            f"follow-up did not reference tool result: {body[:300]}"


def test_multi_tool_choice():
    """Given multiple tools, model should pick the right one for the task."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Use the provided tools."},
            {"role": "user", "content": "Read the file /etc/hostname and tell me what's in it."},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    data = _post(payload)
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")
    assert tool_calls, f"expected a tool call, got: {json.dumps(msg)[:500]}"
    name = tool_calls[0]["function"]["name"]
    assert name in ("read_file", "bash"), f"unexpected tool selected: {name}"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    if name == "read_file":
        assert "/etc/hostname" in args.get("path", "")
    else:
        assert "hostname" in args.get("command", "")
