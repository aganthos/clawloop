"""Tests for SSE and JSON response parsers used by the proxy."""

from __future__ import annotations

import json

from clawloop.proxy_sse import parse_json_response, parse_sse_bytes


def _sse_chunk(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


# ---------------------------------------------------------------------------
# parse_sse_bytes
# ---------------------------------------------------------------------------


class TestParseSSEBytesSimpleText:
    """Two content deltas + usage + [DONE]."""

    def test_simple_text_stream(self) -> None:
        chunks = (
            _sse_chunk(
                {
                    "id": "chatcmpl-1",
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}}],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-1",
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {"content": " world"}}],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-1",
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            )
            + b"data: [DONE]\n\n"
        )

        msg, usage, done = parse_sse_bytes(chunks)

        assert done is True
        assert msg is not None
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert msg["model"] == "gpt-4o"
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class TestParseSSEBytesToolCalls:
    """Tool call deltas: id, function name, incremental arguments."""

    def test_tool_call_deltas(self) -> None:
        chunks = (
            _sse_chunk(
                {
                    "id": "chatcmpl-2",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"lo',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-2",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'cation":"P'},
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-2",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'aris"}'},
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-2",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                }
            )
            + b"data: [DONE]\n\n"
        )

        msg, usage, done = parse_sse_bytes(chunks)

        assert done is True
        assert msg is not None
        assert msg["role"] == "assistant"
        assert msg["content"] is None or msg["content"] == ""
        assert msg["model"] == "gpt-4o-mini"
        assert len(msg["tool_calls"]) == 1

        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"location":"Paris"}'

        assert usage is not None
        assert usage["total_tokens"] == 30

    def test_multiple_tool_calls(self) -> None:
        """Two tool calls in the same stream, using different indices."""
        chunks = (
            _sse_chunk(
                {
                    "id": "chatcmpl-3",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"q":',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-3",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]
                            },
                        }
                    ],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-3",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"id":1}',
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-3",
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
            )
            + b"data: [DONE]\n\n"
        )

        msg, usage, done = parse_sse_bytes(chunks)

        assert done is True
        assert msg is not None
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"q":"hi"}'
        assert msg["tool_calls"][1]["function"]["name"] == "lookup"
        assert msg["tool_calls"][1]["function"]["arguments"] == '{"id":1}'


class TestParseSSEBytesMissingUsage:
    """No usage in any chunk."""

    def test_no_usage(self) -> None:
        chunks = (
            _sse_chunk(
                {
                    "id": "chatcmpl-4",
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
                }
            )
            + _sse_chunk(
                {
                    "id": "chatcmpl-4",
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            + b"data: [DONE]\n\n"
        )

        msg, usage, done = parse_sse_bytes(chunks)

        assert done is True
        assert msg is not None
        assert msg["content"] == "Hi"
        assert usage is None


class TestParseSSEBytesIncomplete:
    """No [DONE] marker -> is_complete=False."""

    def test_incomplete_stream(self) -> None:
        chunks = _sse_chunk(
            {
                "id": "chatcmpl-5",
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}}],
            }
        )

        msg, usage, done = parse_sse_bytes(chunks)

        assert done is False
        assert msg is not None
        assert msg["content"] == "partial"
        assert usage is None


class TestParseSSEBytesEmpty:
    """Empty bytes -> (None, None, False)."""

    def test_empty_bytes(self) -> None:
        msg, usage, done = parse_sse_bytes(b"")
        assert msg is None
        assert usage is None
        assert done is False

    def test_only_whitespace(self) -> None:
        msg, usage, done = parse_sse_bytes(b"  \n\n  ")
        assert msg is None
        assert usage is None
        assert done is False

    def test_only_done(self) -> None:
        """Just [DONE] with no data chunks."""
        msg, usage, done = parse_sse_bytes(b"data: [DONE]\n\n")
        assert msg is None
        assert usage is None
        assert done is True


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Non-streaming JSON response parsing."""

    def test_full_response(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl-99",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        ).encode()

        msg, usage, model = parse_json_response(body)

        assert msg is not None
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert usage == {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
        assert model == "gpt-4o"

    def test_tool_call_response(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl-100",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"q":"test"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }
        ).encode()

        msg, usage, model = parse_json_response(body)

        assert msg is not None
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert model == "gpt-4o"

    def test_missing_usage(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl-101",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()

        msg, usage, model = parse_json_response(body)

        assert msg is not None
        assert msg["content"] == "ok"
        assert usage is None
        assert model == "gpt-4o"


class TestParseJsonResponseMalformed:
    """Malformed JSON -> (None, None, None)."""

    def test_invalid_json(self) -> None:
        msg, usage, model = parse_json_response(b"not json at all")
        assert msg is None
        assert usage is None
        assert model is None

    def test_empty_bytes(self) -> None:
        msg, usage, model = parse_json_response(b"")
        assert msg is None
        assert usage is None
        assert model is None

    def test_valid_json_but_no_choices(self) -> None:
        body = json.dumps({"id": "chatcmpl-err", "model": "gpt-4o"}).encode()
        msg, usage, model = parse_json_response(body)
        assert msg is None
        assert usage is None
        assert model is None
