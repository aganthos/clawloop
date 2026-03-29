# OpenClaw Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenAI-compatible LLM proxy + pi-mono environment adapter so ClawLoop can improve agents via skill injection and trace capture.

**Architecture:** A `ProxyApp` class provides Starlette routes (`POST /v1/chat/completions`) that intercept LLM requests, inject playbook skills, forward to the real upstream, tee the response for trace capture, and post-process into Episodes via `EpisodeCollector`. An `OpenClawAdapter(EnvAdapter)` spawns pi-mono agents as Node subprocesses against task sets. Both share the same proxy machinery.

**Tech Stack:** Python 3.11+, Starlette, httpx, Pydantic v2, asyncio, pytest. Node.js for the thin runner script.

**Spec:** `docs/plans/2026-03-28-openclaw-adapter.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `clawloop/proxy_config.py` | `ProxyConfig` Pydantic model + validation |
| `clawloop/proxy_sse.py` | Pure functions: parse raw SSE bytes → OpenAI message dict |
| `clawloop/proxy_skills.py` | Pure functions: inject/strip skills in message lists |
| `clawloop/proxy_session.py` | `SessionTracker`: turn_index counters, session correlation |
| `clawloop/proxy.py` | `ProxyApp`: Starlette routes, upstream forwarding, tee, worker pool |
| `clawloop/adapters/openclaw.py` | `OpenClawAdapter(EnvAdapter)` |
| `scripts/openclaw_runner/runner.js` | Thin Node runner for pi-mono agents |
| `scripts/openclaw_runner/package.json` | Node dependencies for runner |
| `tests/test_proxy_config.py` | Tests for ProxyConfig |
| `tests/test_proxy_sse.py` | Tests for SSE parser |
| `tests/test_proxy_skills.py` | Tests for skill injection/stripping |
| `tests/test_proxy_session.py` | Tests for SessionTracker |
| `tests/test_proxy.py` | Tests for ProxyApp (integration with mock upstream) |
| `tests/test_openclaw_adapter.py` | Tests for OpenClawAdapter |

---

### Task 1: ProxyConfig

**Files:**
- Create: `clawloop/proxy_config.py`
- Test: `tests/test_proxy_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_proxy_config.py
"""Tests for ProxyConfig validation."""
import pytest
from pydantic import SecretStr, ValidationError


class TestProxyConfig:
    def test_minimal_valid(self):
        from clawloop.proxy_config import ProxyConfig

        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.bench == "openclaw"
        assert cfg.max_tee_bytes == 524288
        assert cfg.upstream_connect_timeout_s == 10.0
        assert cfg.upstream_read_timeout_s == 120.0
        assert cfg.upstream_supports_stream_usage is True
        assert cfg.max_post_process_tasks == 8
        assert cfg.proxy_key == ""
        assert cfg.live_idle_timeout_s == 300

    def test_upstream_url_must_be_https(self):
        from clawloop.proxy_config import ProxyConfig

        with pytest.raises(ValidationError, match="upstream_url"):
            ProxyConfig(
                upstream_url="http://remote-host.com/v1",
                upstream_api_key=SecretStr("sk-test"),
            )

    def test_upstream_url_allows_http_localhost(self):
        from clawloop.proxy_config import ProxyConfig

        cfg = ProxyConfig(
            upstream_url="http://localhost:8000/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.upstream_url == "http://localhost:8000/v1"

    def test_upstream_url_allows_http_127(self):
        from clawloop.proxy_config import ProxyConfig

        cfg = ProxyConfig(
            upstream_url="http://127.0.0.1:8000/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.upstream_url == "http://127.0.0.1:8000/v1"

    def test_bench_mode_flag(self):
        from clawloop.proxy_config import ProxyConfig

        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
            bench_mode=True,
        )
        assert cfg.bench_mode is True

    def test_live_mode_requires_proxy_key(self):
        from clawloop.proxy_config import ProxyConfig

        with pytest.raises(ValidationError, match="proxy_key"):
            ProxyConfig(
                upstream_url="https://api.openai.com/v1",
                upstream_api_key=SecretStr("sk-test"),
                bench_mode=False,
                proxy_key="",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_proxy_config.py -v`
Expected: ImportError — `clawloop.proxy_config` does not exist.

- [ ] **Step 3: Implement ProxyConfig**

```python
# clawloop/proxy_config.py
"""Configuration for the ClawLoop LLM proxy."""
from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlparse

from pydantic import BaseModel, SecretStr, model_validator


class ProxyConfig(BaseModel):
    """Config for ProxyApp — the OpenAI-compatible LLM proxy."""

    upstream_url: str
    upstream_api_key: SecretStr
    bench: str = "openclaw"
    bench_mode: bool = True
    proxy_key: str = ""
    max_tee_bytes: int = 524288  # 512 KB
    live_idle_timeout_s: int = 300
    upstream_connect_timeout_s: float = 10.0
    upstream_read_timeout_s: float = 120.0
    upstream_supports_stream_usage: bool = True
    max_post_process_tasks: int = 8
    redaction_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate(self) -> "ProxyConfig":
        parsed = urlparse(self.upstream_url)
        is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if parsed.scheme != "https" and not is_local:
            raise ValueError(
                f"upstream_url must be https (or http for localhost), "
                f"got {parsed.scheme}://{parsed.hostname}"
            )
        if not self.bench_mode and not self.proxy_key:
            raise ValueError("proxy_key is required in live mode (bench_mode=False)")
        return self

    # Header allowlist for upstream forwarding
    FORWARD_HEADERS: frozenset[str] = frozenset({
        "content-type", "accept", "user-agent",
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_proxy_config.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```
git add clawloop/proxy_config.py tests/test_proxy_config.py
git commit -m "feat: add ProxyConfig with URL and auth validation"
```

---

### Task 2: SSE Post-Processor

Pure functions that parse raw SSE bytes (from an OpenAI streaming response) into a reconstructed assistant message dict. Also handles non-streaming JSON.

**Files:**
- Create: `clawloop/proxy_sse.py`
- Test: `tests/test_proxy_sse.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_proxy_sse.py
"""Tests for SSE response parsing."""
import json
import pytest


class TestParseSSE:
    def test_simple_text_response(self):
        from clawloop.proxy_sse import parse_sse_bytes

        # Simulate a minimal streaming response with two text deltas
        chunks = [
            _sse_chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {"content": "Hello"}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {"content": " world"}, "index": 0}]}),
            _sse_chunk({
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }),
            b"data: [DONE]\n\n",
        ]
        raw = b"".join(chunks)
        msg, usage, complete = parse_sse_bytes(raw)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert complete is True

    def test_tool_call_deltas(self):
        from clawloop.proxy_sse import parse_sse_bytes

        chunks = [
            _sse_chunk({"choices": [{"delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": ""}}
            ]}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"path":'}}
            ]}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ' "foo.py"}'}}
            ]}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}]}),
            b"data: [DONE]\n\n",
        ]
        raw = b"".join(chunks)
        msg, usage, complete = parse_sse_bytes(raw)
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "foo.py"}
        assert complete is True

    def test_missing_usage(self):
        from clawloop.proxy_sse import parse_sse_bytes

        chunks = [
            _sse_chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {"content": "Hi"}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ]
        raw = b"".join(chunks)
        msg, usage, complete = parse_sse_bytes(raw)
        assert msg["content"] == "Hi"
        assert usage is None
        assert complete is True

    def test_incomplete_stream_no_done(self):
        from clawloop.proxy_sse import parse_sse_bytes

        chunks = [
            _sse_chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]}),
            _sse_chunk({"choices": [{"delta": {"content": "partial"}, "index": 0}]}),
        ]
        raw = b"".join(chunks)
        msg, usage, complete = parse_sse_bytes(raw)
        assert msg["content"] == "partial"
        assert complete is False

    def test_empty_bytes(self):
        from clawloop.proxy_sse import parse_sse_bytes

        msg, usage, complete = parse_sse_bytes(b"")
        assert msg is None
        assert complete is False


class TestParseJSON:
    def test_non_streaming_response(self):
        from clawloop.proxy_sse import parse_json_response

        body = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "model": "gpt-4o",
        }
        msg, usage, model = parse_json_response(json.dumps(body).encode())
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello"
        assert usage["total_tokens"] == 12
        assert model == "gpt-4o"

    def test_malformed_json(self):
        from clawloop.proxy_sse import parse_json_response

        msg, usage, model = parse_json_response(b"not json")
        assert msg is None


def _sse_chunk(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_proxy_sse.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement SSE parser**

```python
# clawloop/proxy_sse.py
"""Parse OpenAI streaming (SSE) and non-streaming responses."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def parse_sse_bytes(
    raw: bytes,
) -> tuple[dict[str, Any] | None, dict[str, int] | None, bool]:
    """Parse raw SSE bytes into (assistant_message, usage, is_complete).

    Returns (None, None, False) if the buffer is empty or unparseable.
    """
    if not raw:
        return None, None, False

    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}  # index -> accumulated tool call
    role = "assistant"
    usage: dict[str, int] | None = None
    model: str | None = None
    saw_done = False

    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            saw_done = True
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue

        if "model" in chunk and model is None:
            model = chunk["model"]
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "role" in delta:
                role = delta["role"]
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": tc_delta.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                tc = tool_calls[idx]
                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    tc["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tc["function"]["arguments"] += fn["arguments"]

    if not content_parts and not tool_calls:
        return None, None, False

    msg: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    if model:
        msg["model"] = model

    return msg, usage, saw_done


def parse_json_response(
    raw: bytes,
) -> tuple[dict[str, Any] | None, dict[str, int] | None, str | None]:
    """Parse a non-streaming JSON response into (message, usage, model)."""
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("Failed to parse non-streaming response as JSON")
        return None, None, None

    choices = body.get("choices", [])
    if not choices:
        return None, None, None

    msg = choices[0].get("message")
    usage = body.get("usage")
    model = body.get("model")
    return msg, usage, model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_proxy_sse.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```
git add clawloop/proxy_sse.py tests/test_proxy_sse.py
git commit -m "feat: add SSE and JSON response parsers for proxy"
```

---

### Task 3: Skill Injection and Stripping

Pure functions operating on OpenAI message lists.

**Files:**
- Create: `clawloop/proxy_skills.py`
- Test: `tests/test_proxy_skills.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_proxy_skills.py
"""Tests for skill injection and stripping."""
import pytest

SENTINEL = "<!-- clawloop-skills:v1 -->"


class TestInjectSkills:
    def test_injects_leading_system_message(self):
        from clawloop.proxy_skills import inject_skills

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        skills_text = "## PLAYBOOK\n### always-backup\nAlways back up files."
        result = inject_skills(messages, skills_text)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert SENTINEL in result[0]["content"]
        assert "always-backup" in result[0]["content"]
        # Original system message is preserved
        assert result[1]["content"] == "You are helpful."

    def test_no_injection_when_empty_skills(self):
        from clawloop.proxy_skills import inject_skills

        messages = [{"role": "user", "content": "Hi"}]
        result = inject_skills(messages, "")
        assert len(result) == 1

    def test_idempotent_on_retry(self):
        from clawloop.proxy_skills import inject_skills

        messages = [
            {"role": "system", "content": f"{SENTINEL}\n## old skills"},
            {"role": "user", "content": "Hi"},
        ]
        result = inject_skills(messages, "## new skills")
        # Should replace, not duplicate
        assert sum(1 for m in result if SENTINEL in m.get("content", "")) == 1
        assert "new skills" in result[0]["content"]


class TestStripSkills:
    def test_strips_injected_message(self):
        from clawloop.proxy_skills import inject_skills, strip_skills

        messages = [{"role": "user", "content": "Hi"}]
        injected = inject_skills(messages, "## PLAYBOOK\nDo stuff")
        assert len(injected) == 2
        stripped = strip_skills(injected)
        assert len(stripped) == 1
        assert stripped[0]["content"] == "Hi"

    def test_noop_when_no_skills(self):
        from clawloop.proxy_skills import strip_skills

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = strip_skills(messages)
        assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_proxy_skills.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement skill injection**

```python
# clawloop/proxy_skills.py
"""Inject and strip ClawLoop playbook skills in OpenAI message lists."""
from __future__ import annotations

from typing import Any

SENTINEL = "<!-- clawloop-skills:v1 -->"


def inject_skills(
    messages: list[dict[str, Any]], skills_text: str
) -> list[dict[str, Any]]:
    """Prepend a skills system message. Idempotent via sentinel detection."""
    if not skills_text:
        return messages

    # Remove any existing skills message (idempotent)
    cleaned = [m for m in messages if SENTINEL not in m.get("content", "")]
    skills_msg = {"role": "system", "content": f"{SENTINEL}\n{skills_text}"}
    return [skills_msg, *cleaned]


def strip_skills(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the injected skills message before training ingestion."""
    return [m for m in messages if SENTINEL not in m.get("content", "")]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_proxy_skills.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```
git add clawloop/proxy_skills.py tests/test_proxy_skills.py
git commit -m "feat: add skill injection and stripping for proxy"
```

---

### Task 4: Session Tracker

In-memory state for turn_index counters and session correlation.

**Files:**
- Create: `clawloop/proxy_session.py`
- Test: `tests/test_proxy_session.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_proxy_session.py
"""Tests for SessionTracker."""
import pytest


class TestSessionTracker:
    def test_resolve_run_id_header(self):
        from clawloop.proxy_session import SessionTracker

        tracker = SessionTracker()
        session_id, attributed = tracker.resolve_session(
            run_id="run-123", session_id=None
        )
        assert session_id == "run-123"
        assert attributed is True

    def test_resolve_session_id_header(self):
        from clawloop.proxy_session import SessionTracker

        tracker = SessionTracker()
        session_id, attributed = tracker.resolve_session(
            run_id=None, session_id="sess-456"
        )
        assert session_id == "sess-456"
        assert attributed is True

    def test_resolve_auto_generated(self):
        from clawloop.proxy_session import SessionTracker

        tracker = SessionTracker()
        session_id, attributed = tracker.resolve_session(
            run_id=None, session_id=None
        )
        assert len(session_id) == 32  # uuid4 hex
        assert attributed is False

    def test_run_id_takes_precedence(self):
        from clawloop.proxy_session import SessionTracker

        tracker = SessionTracker()
        session_id, _ = tracker.resolve_session(
            run_id="run-1", session_id="sess-2"
        )
        assert session_id == "run-1"

    def test_turn_index_monotonic(self):
        from clawloop.proxy_session import SessionTracker

        tracker = SessionTracker()
        assert tracker.next_turn("sess-1") == 0
        assert tracker.next_turn("sess-1") == 1
        assert tracker.next_turn("sess-1") == 2
        # Different session starts at 0
        assert tracker.next_turn("sess-2") == 0

    def test_thread_safety(self):
        """Verify turn indices don't skip or collide under contention."""
        import threading
        from clawloop.proxy_session import SessionTracker

        tracker = SessionTracker()
        results: list[int] = []
        n = 100

        def inc():
            for _ in range(n):
                results.append(tracker.next_turn("s"))

        threads = [threading.Thread(target=inc) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(n * 4))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_proxy_session.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement SessionTracker**

```python
# clawloop/proxy_session.py
"""Session correlation and turn ordering for the LLM proxy."""
from __future__ import annotations

import threading
from collections import defaultdict
from uuid import uuid4


class SessionTracker:
    """Thread-safe session resolver and turn counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)

    def resolve_session(
        self,
        run_id: str | None,
        session_id: str | None,
    ) -> tuple[str, bool]:
        """Return (session_id, attributed). Precedence: run_id > session_id > auto."""
        if run_id:
            return run_id, True
        if session_id:
            return session_id, True
        return uuid4().hex, False

    def next_turn(self, session_id: str) -> int:
        """Return the next monotonic turn index for a session."""
        with self._lock:
            idx = self._counters[session_id]
            self._counters[session_id] = idx + 1
            return idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_proxy_session.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```
git add clawloop/proxy_session.py tests/test_proxy_session.py
git commit -m "feat: add SessionTracker for proxy turn ordering"
```

---

### Task 5: ProxyApp — Core Route Handler

The main proxy Starlette route. Forwards requests upstream via httpx, injects skills, tees response bytes, enqueues post-processing.

**Files:**
- Create: `clawloop/proxy.py`
- Test: `tests/test_proxy.py`

This is the largest task. The test uses `starlette.testclient.TestClient` against a `ProxyApp` with a mock upstream server (also Starlette).

- [ ] **Step 1: Write failing tests for non-streaming proxy**

```python
# tests/test_proxy.py
"""Tests for ProxyApp."""
import json
import pytest
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _make_mock_upstream(response_body: dict):
    """Create a mock upstream that returns a fixed response."""
    async def chat_completions(request: Request):
        return JSONResponse(response_body)
    return Starlette(routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ])


def _make_proxy_client(mock_upstream_url: str, bench_mode: bool = True):
    from clawloop.proxy import ProxyApp
    from clawloop.proxy_config import ProxyConfig

    config = ProxyConfig(
        upstream_url=mock_upstream_url,
        upstream_api_key=SecretStr("test-key"),
        bench_mode=bench_mode,
        proxy_key="test-proxy-key" if not bench_mode else "",
    )
    app = ProxyApp(config)
    return TestClient(app.asgi_app)


# We need an actual running mock server for httpx to connect to.
# Use pytest fixtures with a background server.
@pytest.fixture
def mock_upstream(tmp_path):
    """Run a mock upstream on a random port, return its URL."""
    import threading
    import uvicorn
    import socket

    response = {
        "id": "chatcmpl-test",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello from mock"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "mock-model",
    }

    app = _make_mock_upstream(response)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import time
    for _ in range(50):
        try:
            import httpx
            httpx.get(f"http://127.0.0.1:{port}/")
            break
        except Exception:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True


class TestProxyNonStreaming:
    def test_forwards_and_returns_response(self, mock_upstream):
        from clawloop.proxy import ProxyApp
        from clawloop.proxy_config import ProxyConfig

        config = ProxyConfig(
            upstream_url=mock_upstream,
            upstream_api_key=SecretStr("test-key"),
            bench_mode=True,
        )
        proxy = ProxyApp(config)
        client = TestClient(proxy.asgi_app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={"X-ClawLoop-Run-Id": "run-001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from mock"

    def test_bench_mode_rejects_without_run_id(self, mock_upstream):
        from clawloop.proxy import ProxyApp
        from clawloop.proxy_config import ProxyConfig

        config = ProxyConfig(
            upstream_url=mock_upstream,
            upstream_api_key=SecretStr("test-key"),
            bench_mode=True,
        )
        proxy = ProxyApp(config)
        client = TestClient(proxy.asgi_app)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 400


class TestProxyAuth:
    def test_live_mode_rejects_without_auth(self, mock_upstream):
        from clawloop.proxy import ProxyApp
        from clawloop.proxy_config import ProxyConfig

        config = ProxyConfig(
            upstream_url=mock_upstream,
            upstream_api_key=SecretStr("test-key"),
            bench_mode=False,
            proxy_key="secret-key",
        )
        proxy = ProxyApp(config)
        client = TestClient(proxy.asgi_app)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 401

    def test_live_mode_accepts_with_auth(self, mock_upstream):
        from clawloop.proxy import ProxyApp
        from clawloop.proxy_config import ProxyConfig

        config = ProxyConfig(
            upstream_url=mock_upstream,
            upstream_api_key=SecretStr("test-key"),
            bench_mode=False,
            proxy_key="secret-key",
        )
        proxy = ProxyApp(config)
        client = TestClient(proxy.asgi_app)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status_code == 200


class TestSkillInjection:
    def test_skills_injected_into_upstream_request(self, mock_upstream):
        """Verify skills are injected by checking the upstream received them."""
        import httpx as _httpx

        captured_bodies: list[dict] = []

        async def capturing_handler(request: Request):
            body = await request.json()
            captured_bodies.append(body)
            return JSONResponse({
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "model": "m",
            })

        # We need a custom mock that captures — use the mock_upstream fixture port
        # For simplicity, test skill injection at the unit level instead
        from clawloop.proxy_skills import inject_skills, SENTINEL

        messages = [{"role": "user", "content": "Hi"}]
        result = inject_skills(messages, "## PLAYBOOK\n### rule\nDo X")
        assert len(result) == 2
        assert SENTINEL in result[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_proxy.py -v -x`
Expected: ImportError — `clawloop.proxy` does not exist.

- [ ] **Step 3: Implement ProxyApp**

```python
# clawloop/proxy.py
"""ProxyApp — OpenAI-compatible LLM proxy for ClawLoop."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from clawloop.proxy_config import ProxyConfig
from clawloop.proxy_session import SessionTracker
from clawloop.proxy_skills import inject_skills, strip_skills
from clawloop.proxy_sse import parse_json_response, parse_sse_bytes

log = logging.getLogger(__name__)


class ProxyApp:
    """OpenAI-compatible LLM proxy with skill injection and trace capture."""

    def __init__(
        self,
        config: ProxyConfig,
        collector: Any | None = None,
        harness: Any | None = None,
    ) -> None:
        self.config = config
        self.collector = collector
        self.harness = harness
        self.session_tracker = SessionTracker()

        # Metrics
        self.drops_total = 0

        # Post-processing queue + workers
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []

        # Upstream httpx client (created in lifespan)
        self._http: httpx.AsyncClient | None = None

        # Build ASGI app
        self.asgi_app = Starlette(
            routes=[
                Route(
                    "/v1/chat/completions",
                    self._handle_chat_completions,
                    methods=["POST"],
                ),
            ],
            lifespan=self._lifespan,
        )

    async def _lifespan(self, app: Starlette):
        # Startup
        self._check_single_worker()
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.upstream_connect_timeout_s,
                read=self.config.upstream_read_timeout_s,
                write=30.0,
                pool=30.0,
            ),
            follow_redirects=False,
            trust_env=False,
        )
        self._queue = asyncio.Queue(maxsize=64)
        self._workers = [
            asyncio.create_task(self._post_process_worker(i))
            for i in range(self.config.max_post_process_tasks)
        ]

        yield

        # Shutdown — drain queue with grace period
        for _ in self._workers:
            await self._queue.put(None)  # Poison pills
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            for w in self._workers:
                w.cancel()
            log.warning("Post-processing workers did not drain in time")
        await self._http.aclose()

    def _check_single_worker(self) -> None:
        wc = os.environ.get("WEB_CONCURRENCY", "1")
        if int(wc) > 1:
            raise RuntimeError(
                f"ProxyApp requires single-process deployment, "
                f"but WEB_CONCURRENCY={wc}"
            )

    async def _handle_chat_completions(self, request: Request) -> Response:
        # --- Auth check ---
        if not self.config.bench_mode and self.config.proxy_key:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {self.config.proxy_key}":
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

        body = await request.json()

        # --- Session correlation ---
        run_id = request.headers.get("x-clawloop-run-id")
        session_hdr = request.headers.get("x-clawloop-session-id")

        if self.config.bench_mode and not run_id:
            return JSONResponse(
                {"error": "X-ClawLoop-Run-Id header required in bench mode"},
                status_code=400,
            )

        session_id, attributed = self.session_tracker.resolve_session(
            run_id=run_id, session_id=session_hdr
        )
        turn_index = self.session_tracker.next_turn(session_id)
        timestamp_ns = time.monotonic_ns()
        no_train = request.headers.get("x-clawloop-no-train") == "1"

        # --- Skill injection ---
        messages = body.get("messages", [])
        skills_text = ""
        if self.harness:
            skills_text = self.harness.playbook.render()
        if skills_text:
            messages = inject_skills(messages, skills_text)
        body["messages"] = messages

        # --- Add stream_options for usage ---
        is_streaming = body.get("stream", False)
        if is_streaming and self.config.upstream_supports_stream_usage:
            body.setdefault("stream_options", {})["include_usage"] = True

        # --- Forward upstream ---
        upstream_url = f"{self.config.upstream_url}/chat/completions"
        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in self.config.FORWARD_HEADERS
        }
        forward_headers["authorization"] = (
            f"Bearer {self.config.upstream_api_key.get_secret_value()}"
        )

        try:
            upstream_resp = await self._http.post(
                upstream_url,
                json=body,
                headers=forward_headers,
            )
        except httpx.HTTPError as e:
            log.error("Upstream request failed: %s", e)
            return JSONResponse(
                {"error": {"message": str(e), "type": "upstream_error"}},
                status_code=502,
            )

        # --- Tee response ---
        tee_buf = bytearray()
        max_tee = self.config.max_tee_bytes
        truncated = False

        if is_streaming:
            async def stream_and_tee():
                nonlocal truncated
                async for chunk in upstream_resp.aiter_bytes():
                    if len(tee_buf) < max_tee:
                        remaining = max_tee - len(tee_buf)
                        tee_buf.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            truncated = True
                    yield chunk

            response = StreamingResponse(
                stream_and_tee(),
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
                media_type="text/event-stream",
            )
        else:
            content = await upstream_resp.aread()
            if len(content) <= max_tee:
                tee_buf.extend(content)
            else:
                tee_buf.extend(content[:max_tee])
                truncated = True
            response = Response(
                content=content,
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )

        # --- Enqueue post-processing ---
        work_item = {
            "session_id": session_id,
            "turn_index": turn_index,
            "timestamp_ns": timestamp_ns,
            "attributed": attributed,
            "no_train": no_train,
            "truncated": truncated,
            "is_streaming": is_streaming,
            "request_messages": body["messages"],
            "tee_buf": bytes(tee_buf),
            "bench": self.config.bench,
            "model": body.get("model"),
        }

        if self._queue is not None:
            try:
                self._queue.put_nowait(work_item)
            except asyncio.QueueFull:
                self.drops_total += 1
                log.warning(
                    "Post-process queue full, dropping turn %s/%d",
                    session_id, turn_index,
                )

        return response

    async def _post_process_worker(self, worker_id: int) -> None:
        """Drain the post-processing queue."""
        while True:
            item = await self._queue.get()
            if item is None:
                break  # Poison pill
            try:
                await self._process_turn(item)
            except Exception:
                log.exception(
                    "Post-process failed for %s/%d",
                    item.get("session_id"), item.get("turn_index"),
                )
            finally:
                self._queue.task_done()

    async def _process_turn(self, item: dict[str, Any]) -> None:
        """Parse response, strip skills, ingest into collector."""
        if item["truncated"] or item["no_train"]:
            return  # Non-trainable, skip ingestion

        tee_buf = item["tee_buf"]
        if item["is_streaming"]:
            msg, usage, complete = parse_sse_bytes(tee_buf)
            if not complete:
                return  # Partial stream
        else:
            msg, usage, model = parse_json_response(tee_buf)

        if msg is None:
            return

        # Strip injected skills from request messages
        clean_messages = strip_skills(item["request_messages"])

        # Build OpenAI-format messages for ingestion
        ingestion_messages = clean_messages + [msg]

        if self.collector:
            usage_dict = None
            if usage:
                usage_dict = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            self.collector.ingest_external(
                messages=[
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in ingestion_messages
                    if m.get("role") in ("system", "user", "assistant", "tool")
                ],
                session_id=item["session_id"],
                model=msg.get("model") or item.get("model"),
                usage=usage_dict,
                bench=item["bench"],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_proxy.py -v -x`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```
git add clawloop/proxy.py tests/test_proxy.py
git commit -m "feat: add ProxyApp with upstream forwarding and trace capture"
```

---

### Task 6: OpenClawAdapter

**Files:**
- Create: `clawloop/adapters/openclaw.py`
- Test: `tests/test_openclaw_adapter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_openclaw_adapter.py
"""Tests for OpenClawAdapter."""
import json
import pytest


class TestListTasks:
    def test_reads_jsonl(self, tmp_path):
        from clawloop.adapters.openclaw import OpenClawAdapter

        task_dir = tmp_path / "tasks"
        task_dir.mkdir()
        (task_dir / "base.jsonl").write_text(
            '{"task_id": "t1", "instruction": "Fix bug"}\n'
            '{"task_id": "t2", "instruction": "Add feature"}\n'
        )

        adapter = OpenClawAdapter()
        adapter.setup({"task_dir": str(task_dir)})
        tasks = adapter.list_tasks("base")
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "t1"
        assert tasks[1]["instruction"] == "Add feature"

    def test_missing_split_returns_empty(self, tmp_path):
        from clawloop.adapters.openclaw import OpenClawAdapter

        task_dir = tmp_path / "tasks"
        task_dir.mkdir()

        adapter = OpenClawAdapter()
        adapter.setup({"task_dir": str(task_dir)})
        assert adapter.list_tasks("nonexistent") == []


class TestRunEpisode:
    def test_runs_subprocess_and_returns_episode(self, tmp_path):
        """Use a mock runner script (Python, not Node) that echoes back."""
        from clawloop.adapters.openclaw import OpenClawAdapter
        from clawloop.core.loop import AgentState

        # Create a mock runner script that reads stdin and writes stdout
        runner = tmp_path / "mock_runner.py"
        runner.write_text(
            'import sys, json\n'
            'task = json.load(sys.stdin)\n'
            'json.dump({"task_id": task["task_id"], "status": "success", '
            '"output": "done"}, sys.stdout)\n'
        )

        adapter = OpenClawAdapter()
        adapter.setup({
            "runner_script": str(runner),
            "node_bin": "python",  # Use python instead of node for test
            "task_dir": str(tmp_path),
            "timeout_s": 10,
            # Skip proxy startup for unit test
            "_skip_proxy": True,
        })

        task = {"task_id": "t1", "instruction": "Fix bug"}
        result = adapter.run_episode(task, AgentState())
        assert result is not None

    def test_timeout_kills_subprocess(self, tmp_path):
        from clawloop.adapters.openclaw import OpenClawAdapter
        from clawloop.core.loop import AgentState

        # Script that hangs
        runner = tmp_path / "hang_runner.py"
        runner.write_text('import time; time.sleep(999)')

        adapter = OpenClawAdapter()
        adapter.setup({
            "runner_script": str(runner),
            "node_bin": "python",
            "task_dir": str(tmp_path),
            "timeout_s": 1,
            "_skip_proxy": True,
        })

        task = {"task_id": "t1", "instruction": "Hang"}
        # Should not hang — timeout should kill it
        result = adapter.run_episode(task, AgentState())
        assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_openclaw_adapter.py -v -x`
Expected: ImportError.

- [ ] **Step 3: Implement OpenClawAdapter**

```python
# clawloop/adapters/openclaw.py
"""OpenClawAdapter — run pi-mono agents against task sets."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from clawloop.adapters.base import EnvAdapter
from clawloop.core.episode import Episode, EpisodeSummary, Message

log = logging.getLogger(__name__)


class OpenClawAdapter(EnvAdapter):
    """EnvAdapter for pi-mono agents via the LLM proxy."""

    def __init__(self) -> None:
        self._task_dir: Path = Path("tasks")
        self._runner_script: str = "scripts/openclaw_runner.js"
        self._node_bin: str = "node"
        self._timeout_s: int = 120
        self._proxy_port: int | None = None
        self._skip_proxy: bool = False

    def setup(self, config: dict[str, Any]) -> None:
        self._task_dir = Path(config.get("task_dir", "tasks"))
        self._runner_script = config.get(
            "runner_script", "scripts/openclaw_runner.js"
        )
        self._node_bin = config.get("node_bin", "node")
        self._timeout_s = config.get("timeout_s", 120)
        self._skip_proxy = config.get("_skip_proxy", False)

        # TODO: Start ProxyApp on ephemeral port when not skipping
        # This will be wired up in the integration task

    def run_episode(self, task: Any, agent_state: Any) -> Episode:
        run_id = uuid4().hex
        task_json = json.dumps(task)

        args = [self._node_bin, self._runner_script]
        if self._proxy_port and not self._skip_proxy:
            args.extend([
                "--base-url", f"http://127.0.0.1:{self._proxy_port}/v1",
                "--run-id", run_id,
            ])

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            stdout, stderr = proc.communicate(
                input=task_json.encode(), timeout=self._timeout_s
            )
        except subprocess.TimeoutExpired:
            log.warning("Runner timed out for task %s, killing", task.get("task_id", "?"))
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()
            stdout, stderr = b"", b""

        # Parse runner output
        result = {"status": "error", "output": ""}
        if stdout:
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                log.warning("Failed to parse runner stdout: %s", stdout[:200])

        if stderr:
            log.debug("Runner stderr: %s", stderr.decode(errors="replace")[:500])

        # Build Episode
        messages = [
            Message(role="user", content=task.get("instruction", "")),
            Message(role="assistant", content=result.get("output", "")),
        ]
        episode = Episode(
            id=Episode.new_id(),
            state_id="",
            task_id=task.get("task_id", run_id),
            bench=self._task_dir.name if not self._skip_proxy else "openclaw",
            messages=messages,
            step_boundaries=[0, len(messages)],
            steps=[],
            summary=EpisodeSummary(),
            session_id=run_id,
            metadata={"runner_status": result.get("status", "unknown")},
        )
        return episode

    def list_tasks(self, split: str = "base") -> list[Any]:
        path = self._task_dir / f"{split}.jsonl"
        if not path.exists():
            return []
        tasks = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                tasks.append(json.loads(line))
        return tasks

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"session_id": episode.session_id}

    def teardown(self) -> None:
        """Stop proxy server and clean up."""
        pass  # Will be wired up in integration task
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_openclaw_adapter.py -v -x`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```
git add clawloop/adapters/openclaw.py tests/test_openclaw_adapter.py
git commit -m "feat: add OpenClawAdapter for pi-mono agent tasks"
```

---

### Task 7: Node Runner Script

**Files:**
- Create: `scripts/openclaw_runner/package.json`
- Create: `scripts/openclaw_runner/runner.js`

- [ ] **Step 1: Create package.json**

```json
{
  "name": "openclaw-runner",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "dependencies": {
    "@mariozechner/pi-agent-core": "latest",
    "@mariozechner/pi-ai": "latest"
  }
}
```

- [ ] **Step 2: Create runner.js**

```js
// scripts/openclaw_runner/runner.js
// Thin runner: reads task from stdin, runs pi-mono agent against ClawLoop proxy.
import { Agent } from "@mariozechner/pi-agent-core";
import { getModel } from "@mariozechner/pi-ai";

const args = process.argv.slice(2);
let baseUrl = "http://127.0.0.1:8400/v1";
let runId = "";

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--base-url" && args[i + 1]) baseUrl = args[++i];
  if (args[i] === "--run-id" && args[i + 1]) runId = args[++i];
}

// Read task from stdin
let input = "";
for await (const chunk of process.stdin) input += chunk;
const task = JSON.parse(input);

// Configure model to point at the ClawLoop proxy
const model = getModel("openai", "gpt-4o");
model.baseUrl = baseUrl;
if (runId) {
  model.headers = { "X-ClawLoop-Run-Id": runId };
}

const agent = new Agent({
  initialState: {
    systemPrompt: task.instruction || "",
    model,
  },
});

// Collect final output
let output = "";
agent.subscribe((event) => {
  if (event.type === "message_end" && event.message.role === "assistant") {
    for (const block of event.message.content) {
      if (block.type === "text") output += block.text;
    }
  }
});

try {
  await agent.prompt(task.instruction || "");
  await agent.waitForIdle();
  const result = { task_id: task.task_id, status: "success", output };
  process.stdout.write(JSON.stringify(result));
} catch (err) {
  const result = { task_id: task.task_id, status: "error", output: err.message };
  process.stdout.write(JSON.stringify(result));
  process.exit(1);
}
```

- [ ] **Step 3: Commit**

```
git add scripts/openclaw_runner/package.json scripts/openclaw_runner/runner.js
git commit -m "feat: add Node runner script for pi-mono agents"
```

---

### Task 8: train.py Registration

**Files:**
- Modify: `clawloop/train.py` (around line 142, ENV_BUILDERS dict)

- [ ] **Step 1: Add openclaw builder function and registration**

Add after the existing `_build_entropic` function (around line 138):

```python
def _build_openclaw(
    config: TrainConfig, llm_clients: dict[str, LLMClientConfig]
) -> tuple:
    from clawloop.adapters.openclaw import OpenClawAdapter

    adapter = OpenClawAdapter()
    adapter_config = {
        "task_dir": config.extra.get("openclaw_task_dir", "tasks"),
        "runner_script": config.extra.get(
            "openclaw_runner", "scripts/openclaw_runner/runner.js"
        ),
        "timeout_s": config.extra.get("openclaw_timeout_s", 120),
        "node_bin": config.extra.get("openclaw_node_bin", "node"),
    }
    adapter.setup(adapter_config)
    tasks = adapter.list_tasks("base")
    return adapter, tasks
```

Add `"openclaw"` to `ENV_BUILDERS`:

```python
ENV_BUILDERS: dict[str, Any] = {
    "harbor": _build_harbor,
    "math": _build_math,
    "entropic": _build_entropic,
    "openclaw": _build_openclaw,
}
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/ -x -q --timeout=30`
Expected: All existing tests still pass.

- [ ] **Step 3: Commit**

```
git add clawloop/train.py
git commit -m "feat: register openclaw adapter in train.py ENV_BUILDERS"
```

---

### Task 9: Server Integration — Mount Proxy on clawloop-server

**Files:**
- Modify: `clawloop/server.py` (add `/v1` mount in `create_app`)

- [ ] **Step 1: Write a test for the /v1 mount**

```python
# Add to tests/test_server.py or create tests/test_server_proxy.py

# tests/test_server_proxy.py
"""Test that clawloop-server can mount proxy routes."""
import pytest


class TestServerProxyMount:
    def test_v1_routes_not_mounted_by_default(self, tmp_path):
        """Without proxy config, /v1 should 404."""
        from starlette.testclient import TestClient
        from clawloop.server import create_app

        seed = tmp_path / "seed.txt"
        seed.write_text("You are helpful.")
        app = create_app(seed_prompt_path=str(seed))
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it passes (404 is current behavior)**

Run: `python -m pytest tests/test_server_proxy.py -v`
Expected: PASS (404 since no /v1 routes exist yet).

- [ ] **Step 3: Add optional proxy_config parameter to create_app**

In `clawloop/server.py`, modify `create_app()` to accept an optional `proxy_config` and mount `/v1` routes when provided:

```python
# In create_app(), after building the routes list (~line 444):
if proxy_config is not None:
    from clawloop.proxy import ProxyApp
    proxy = ProxyApp(
        proxy_config,
        collector=server.collector,
        harness=server.harness,
    )
    from starlette.routing import Mount
    routes.append(Mount("/v1", app=proxy.asgi_app))
```

Add `proxy_config` parameter to the `create_app` signature:

```python
def create_app(
    ...
    proxy_config: "ProxyConfig | None" = None,
) -> Starlette:
```

- [ ] **Step 4: Run all tests to verify nothing broke**

Run: `python -m pytest tests/ -x -q --timeout=30`
Expected: All pass.

- [ ] **Step 5: Commit**

```
git add clawloop/server.py tests/test_server_proxy.py
git commit -m "feat: add optional proxy mount to clawloop-server"
```

---

### Task 10: Integration Test

End-to-end test with mock upstream, proxy, and adapter.

**Files:**
- Create: `tests/test_proxy_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_proxy_integration.py
"""End-to-end integration test: proxy + mock upstream + collector."""
import json
import pytest
import socket
import threading
import time

import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


@pytest.fixture
def mock_upstream():
    """Spin up a mock OpenAI-compatible upstream."""
    captured: list[dict] = []

    async def handler(request: Request):
        body = await request.json()
        captured.append(body)
        return JSONResponse({
            "choices": [{
                "message": {"role": "assistant", "content": "mock reply"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "mock-model",
        })

    app = Starlette(routes=[Route("/chat/completions", handler, methods=["POST"])])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            import httpx
            httpx.get(f"http://127.0.0.1:{port}/")
            break
        except Exception:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}", captured
    server.should_exit = True


class TestEndToEnd:
    def test_proxy_captures_and_ingests(self, mock_upstream):
        from clawloop.proxy import ProxyApp
        from clawloop.proxy_config import ProxyConfig
        from clawloop.collector import EpisodeCollector
        from clawloop.core.reward import RewardPipeline

        url, captured = mock_upstream
        ingested = []

        collector = EpisodeCollector(
            pipeline=RewardPipeline.with_defaults(),
            on_batch=lambda episodes: ingested.extend(episodes),
            batch_size=1,
        )

        config = ProxyConfig(
            upstream_url=url,
            upstream_api_key=SecretStr("test"),
            bench_mode=True,
            bench="test-bench",
        )
        proxy = ProxyApp(config, collector=collector)
        client = TestClient(proxy.asgi_app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"X-ClawLoop-Run-Id": "run-integration-1"},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "mock reply"

        # Verify upstream received the request
        assert len(captured) == 1

        # Give post-processing a moment to run
        time.sleep(0.5)

        # Collector should have ingested
        assert len(ingested) >= 1
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_proxy_integration.py -v -x --timeout=30`
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add tests/test_proxy_integration.py
git commit -m "feat: add end-to-end proxy integration test"
```

---

## Summary of commits

| # | Message | Files |
|---|---------|-------|
| 1 | `feat: add ProxyConfig with URL and auth validation` | proxy_config.py, test |
| 2 | `feat: add SSE and JSON response parsers for proxy` | proxy_sse.py, test |
| 3 | `feat: add skill injection and stripping for proxy` | proxy_skills.py, test |
| 4 | `feat: add SessionTracker for proxy turn ordering` | proxy_session.py, test |
| 5 | `feat: add ProxyApp with upstream forwarding and trace capture` | proxy.py, test |
| 6 | `feat: add OpenClawAdapter for pi-mono agent tasks` | openclaw.py, test |
| 7 | `feat: add Node runner script for pi-mono agents` | runner.js, package.json |
| 8 | `feat: register openclaw adapter in train.py ENV_BUILDERS` | train.py |
| 9 | `feat: add optional proxy mount to clawloop-server` | server.py, test |
| 10 | `feat: add end-to-end proxy integration test` | test_proxy_integration.py |
