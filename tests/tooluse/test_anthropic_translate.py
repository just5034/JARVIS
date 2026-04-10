"""Unit tests for the Anthropic ↔ OpenAI translation layer.

These run locally with no server needed — they test pure JSON translation.
"""

from __future__ import annotations

import json

from jarvis.tooluse.anthropic_translate import (
    StreamTranslator,
    translate_request,
    translate_response,
)


class TestTranslateRequest:
    """Anthropic Messages → OpenAI Chat Completions request translation."""

    def test_simple_text(self):
        body = {
            "model": "qwen3.5-27b",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }
        result = translate_request(body)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "Hello"}
        assert result["max_tokens"] == 1024

    def test_system_as_blocks(self):
        body = {
            "model": "qwen3.5-27b",
            "system": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        result = translate_request(body)
        assert result["messages"][0]["content"] == "Part one.\nPart two."

    def test_tools_translated(self):
        body = {
            "model": "qwen3.5-27b",
            "messages": [{"role": "user", "content": "Read a file"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            "max_tokens": 1024,
        }
        result = translate_request(body)
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "read_file"
        assert tool["function"]["parameters"]["properties"]["path"]["type"] == "string"

    def test_tool_choice_auto(self):
        body = {
            "model": "qwen3.5-27b",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
            "max_tokens": 1024,
        }
        result = translate_request(body)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        body = {
            "model": "qwen3.5-27b",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
            "max_tokens": 1024,
        }
        result = translate_request(body)
        assert result["tool_choice"] == "required"

    def test_tool_choice_specific(self):
        body = {
            "model": "qwen3.5-27b",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "read_file"},
            "max_tokens": 1024,
        }
        result = translate_request(body)
        assert result["tool_choice"]["function"]["name"] == "read_file"

    def test_assistant_tool_use_blocks(self):
        """Assistant messages with tool_use blocks → OpenAI tool_calls."""
        body = {
            "model": "qwen3.5-27b",
            "messages": [
                {"role": "user", "content": "Read /tmp"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me read that."},
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "read_file",
                            "input": {"path": "/tmp"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": "file contents here",
                        }
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = translate_request(body)
        msgs = result["messages"]

        # user message
        assert msgs[0] == {"role": "user", "content": "Read /tmp"}

        # assistant with tool_calls
        asst = msgs[1]
        assert asst["role"] == "assistant"
        assert asst["content"] == "Let me read that."
        assert len(asst["tool_calls"]) == 1
        tc = asst["tool_calls"][0]
        assert tc["id"] == "toolu_abc123"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp"}

        # tool result
        tool_msg = msgs[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_abc123"
        assert tool_msg["content"] == "file contents here"

    def test_ignores_anthropic_extras(self):
        """betas, cache_control, thinking, metadata should not cause errors."""
        body = {
            "model": "qwen3.5-27b",
            "system": "Hi",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "metadata": {"user_id": "test"},
            "betas": ["some-beta-header"],
            "output_config": {"effort": "high"},
            "context_management": {"type": "auto"},
        }
        result = translate_request(body)
        # Should succeed without error, extras silently dropped
        assert result["messages"][0]["content"] == "Hi"
        assert "thinking" not in result
        assert "betas" not in result


class TestTranslateResponse:
    """OpenAI Chat Completions → Anthropic Messages response translation."""

    def test_text_response(self):
        openai = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = translate_response(openai, model="qwen3.5-27b")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_call_response(self):
        openai = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/foo.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }
        result = translate_response(openai, model="qwen3.5-27b")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_abc"
        assert block["name"] == "read_file"
        assert block["input"] == {"path": "/tmp/foo.txt"}

    def test_mixed_text_and_tool(self):
        openai = {
            "id": "chatcmpl-789",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll read that file.",
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/etc/hostname"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        result = translate_response(openai, model="qwen3.5-27b")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"


class TestStreamTranslator:
    """OpenAI streaming chunks → Anthropic SSE events."""

    def test_text_streaming(self):
        t = StreamTranslator(model="qwen3.5-27b")

        events = t.feed_chunk({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
        })

        # Should get: message_start, content_block_start(text), content_block_delta(text)
        assert any("message_start" in e for e in events)
        assert any("content_block_start" in e for e in events)
        assert any("text_delta" in e for e in events)
        assert any("Hello" in e for e in events)

    def test_tool_call_streaming(self):
        t = StreamTranslator(model="qwen3.5-27b")

        # First chunk: tool call start with name
        events1 = t.feed_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "function": {"name": "read_file", "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
        })
        assert any("content_block_start" in e for e in events1)
        assert any("tool_use" in e for e in events1)
        assert any("read_file" in e for e in events1)

        # Second chunk: argument fragment
        events2 = t.feed_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"path":'},
                    }]
                },
                "finish_reason": None,
            }]
        })
        assert any("input_json_delta" in e for e in events2)

        # Third chunk: finish
        events3 = t.feed_chunk({
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}]
        })
        assert any("content_block_stop" in e for e in events3)
        assert any("message_delta" in e for e in events3)
        assert any("tool_use" in e for e in events3)  # stop_reason
        assert any("message_stop" in e for e in events3)

    def test_text_then_tool(self):
        """Text followed by tool call should produce separate blocks."""
        t = StreamTranslator(model="qwen3.5-27b")

        # Text chunk
        events1 = t.feed_chunk({
            "choices": [{"delta": {"content": "Let me check."}, "finish_reason": None}]
        })
        text_starts = [e for e in events1 if "content_block_start" in e and "\"text\"" in e]
        assert len(text_starts) == 1

        # Tool chunk — should close text block and open tool block
        events2 = t.feed_chunk({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "bash", "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
        })
        assert any("content_block_stop" in e for e in events2)
        tool_starts = [e for e in events2 if "content_block_start" in e and "tool_use" in e]
        assert len(tool_starts) == 1
