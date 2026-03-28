"""End-to-end integration test: mock upstream -> proxy -> EpisodeCollector."""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

import pytest
import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig
from clawloop.proxy_skills import SENTINEL

# ---------------------------------------------------------------------------
# Harness stub for skill injection tests
# ---------------------------------------------------------------------------


@dataclass
class _StubPlaybook:
    """Minimal stub that satisfies harness.playbook.render()."""
    text: str

    def render(self) -> str:
        return self.text


@dataclass
class _StubHarness:
    """Minimal harness stub with a .playbook.render() method."""
    playbook: _StubPlaybook


def _make_harness(skills_text: str = "Always be helpful.") -> _StubHarness:
    return _StubHarness(playbook=_StubPlaybook(text=skills_text))


# ---------------------------------------------------------------------------
# Helper: start a real uvicorn server on an ephemeral port
# ---------------------------------------------------------------------------


def _start_server(app: Starlette) -> tuple[str, uvicorn.Server, threading.Thread]:
    """Start *app* on an ephemeral port, return (url, server, thread)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"),
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("Server did not start in time")

    return f"http://127.0.0.1:{port}", server, thread


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_upstream():
    """Spin up a mock OpenAI-compatible upstream on an ephemeral port."""
    captured: list[dict] = []

    async def handler(request: Request) -> JSONResponse:
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

    app = Starlette(
        routes=[Route("/chat/completions", handler, methods=["POST"])],
    )
    url, server, thread = _start_server(app)
    yield url, captured
    server.should_exit = True
    thread.join(timeout=3)


@pytest.fixture()
def mock_streaming_upstream():
    """Upstream that returns SSE streaming chunks."""
    captured: list[dict] = []

    async def handler(request: Request) -> StreamingResponse:
        body = await request.json()
        captured.append(body)

        async def generate():
            yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello'}, 'index': 0}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': ' world'}, 'index': 0}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    app = Starlette(
        routes=[Route("/chat/completions", handler, methods=["POST"])],
    )
    url, server, thread = _start_server(app)
    yield url, captured
    server.should_exit = True
    thread.join(timeout=3)


@pytest.fixture()
def mock_error_upstream():
    """Upstream that returns 429 rate-limit error."""
    async def handler(request: Request) -> JSONResponse:
        return JSONResponse(
            {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            status_code=429,
        )

    app = Starlette(
        routes=[Route("/chat/completions", handler, methods=["POST"])],
    )
    url, server, thread = _start_server(app)
    yield url
    server.should_exit = True
    thread.join(timeout=3)


def _wait_for_ingestion(ingested: list, *, count: int = 1, timeout: float = 5.0) -> None:
    """Poll until *ingested* has at least *count* items (up to *timeout* seconds)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(ingested) >= count:
            return
        time.sleep(0.1)


def _assert_no_ingestion(ingested: list, *, settle: float = 0.3, polls: int = 5) -> None:
    """Assert that nothing was ingested after giving the queue time to settle.

    Polls *polls* times over *settle* seconds, asserting zero ingestion each
    time.  This avoids the race in a single ``time.sleep`` — if a bug caused
    ingestion it would almost certainly appear within the polling window.
    """
    interval = settle / polls
    for _ in range(polls):
        time.sleep(interval)
        assert len(ingested) == 0, (
            f"Expected no ingestion but got {len(ingested)} item(s)"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_proxy_captures_and_ingests(self, mock_upstream):
        """Full round-trip: client -> proxy -> upstream -> collector ingest."""
        url, captured = mock_upstream
        ingested: list = []

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

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-integration-1"},
            )

            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "mock reply"

            # Verify upstream received the request
            assert len(captured) == 1

            # Give post-processing workers time to ingest
            _wait_for_ingestion(ingested)

        # Collector should have ingested at least one episode
        assert len(ingested) >= 1

        ep = ingested[0]
        assert ep.bench == "test-bench"
        # The episode should contain the user message and the assistant reply
        roles = [m.role for m in ep.messages]
        assert "user" in roles
        assert "assistant" in roles

    # -----------------------------------------------------------------------
    # Test 1: Skill injection reaches upstream
    # -----------------------------------------------------------------------

    def test_skill_injection_reaches_upstream(self, mock_upstream):
        """Harness playbook skills are injected and forwarded to upstream."""
        url, captured = mock_upstream
        harness = _make_harness("Always be helpful.")

        config = ProxyConfig(
            upstream_url=url,
            upstream_api_key=SecretStr("test"),
            bench_mode=True,
            bench="test-bench",
        )
        proxy = ProxyApp(config, harness=harness)

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-skills-1"},
            )

            assert resp.status_code == 200

        # The upstream should have received the request with skills injected
        assert len(captured) == 1
        upstream_messages = captured[0]["messages"]

        # First message should be the injected skills system message
        first_msg = upstream_messages[0]
        assert first_msg["role"] == "system"
        assert SENTINEL in first_msg["content"]
        assert "Always be helpful." in first_msg["content"]

        # Original user message should follow
        assert upstream_messages[1]["role"] == "user"
        assert upstream_messages[1]["content"] == "Hi"

    # -----------------------------------------------------------------------
    # Test 2: Collector ingests with skills stripped
    # -----------------------------------------------------------------------

    def test_collector_ingests_with_skills_stripped(self, mock_upstream):
        """After proxying, ingested episodes have skills stripped from messages."""
        url, captured = mock_upstream
        ingested: list = []
        harness = _make_harness("Secret skill text.")

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
        proxy = ProxyApp(config, collector=collector, harness=harness)

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello skills"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-skills-strip-1"},
            )

            assert resp.status_code == 200

            # Upstream DID receive the skills
            assert len(captured) == 1
            assert SENTINEL in captured[0]["messages"][0]["content"]

            _wait_for_ingestion(ingested)

        assert len(ingested) >= 1
        ep = ingested[0]

        # Ingested messages must NOT contain the sentinel
        for msg in ep.messages:
            assert SENTINEL not in (msg.content or ""), (
                f"Skills sentinel found in ingested message: {msg.content!r}"
            )

        # But they must contain the user and assistant messages
        roles = [m.role for m in ep.messages]
        assert "user" in roles
        assert "assistant" in roles
        contents = [m.content for m in ep.messages]
        assert "Hello skills" in contents
        assert "mock reply" in contents

    # -----------------------------------------------------------------------
    # Test 3: X-ClawLoop-No-Train header skips ingestion
    # -----------------------------------------------------------------------

    def test_no_train_header_skips_ingestion(self, mock_upstream):
        """X-ClawLoop-No-Train: 1 causes response to pass but skips ingestion."""
        url, _captured = mock_upstream
        ingested: list = []

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

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "No train please"}],
                },
                headers={
                    "X-ClawLoop-Run-Id": "run-notrain-1",
                    "X-ClawLoop-No-Train": "1",
                },
            )

            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "mock reply"

            # Deterministic: poll briefly to confirm nothing was ingested
            _assert_no_ingestion(ingested)

        assert len(ingested) == 0, "No-train request should not be ingested"

    # -----------------------------------------------------------------------
    # Test 4: Streaming SSE end-to-end
    # -----------------------------------------------------------------------

    def test_streaming_sse_end_to_end(self, mock_streaming_upstream):
        """Streaming SSE chunks flow through proxy and collector ingests assembled content."""
        url, captured = mock_streaming_upstream
        ingested: list = []

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

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Stream me"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-stream-1"},
            )

            assert resp.status_code == 200
            body_text = resp.text
            assert "Hello" in body_text
            assert "world" in body_text
            assert "[DONE]" in body_text

            _wait_for_ingestion(ingested)

        assert len(ingested) >= 1
        ep = ingested[0]
        # The assembled assistant message should contain "Hello world"
        assistant_msgs = [m for m in ep.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert "Hello world" in (assistant_msgs[0].content or "")

    # -----------------------------------------------------------------------
    # Test 5: Upstream error passthrough
    # -----------------------------------------------------------------------

    def test_upstream_error_passthrough(self, mock_error_upstream):
        """Upstream 429 is passed through to the client unchanged."""
        url = mock_error_upstream

        config = ProxyConfig(
            upstream_url=url,
            upstream_api_key=SecretStr("test"),
            bench_mode=True,
            bench="test-bench",
        )
        proxy = ProxyApp(config)

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Trigger error"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-error-1"},
            )

            assert resp.status_code == 429
            body = resp.json()
            assert body["error"]["type"] == "rate_limit_error"
            assert "Rate limit exceeded" in body["error"]["message"]

    # -----------------------------------------------------------------------
    # Test 6: stream_options.include_usage forwarded
    # -----------------------------------------------------------------------

    def test_stream_options_include_usage_forwarded(self, mock_streaming_upstream):
        """When upstream_supports_stream_usage=True, stream_options.include_usage is set."""
        url, captured = mock_streaming_upstream

        config = ProxyConfig(
            upstream_url=url,
            upstream_api_key=SecretStr("test"),
            bench_mode=True,
            bench="test-bench",
            upstream_supports_stream_usage=True,
        )
        proxy = ProxyApp(config)

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Usage test"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-usage-1"},
            )

            assert resp.status_code == 200

        assert len(captured) == 1
        upstream_body = captured[0]
        assert upstream_body.get("stream_options", {}).get("include_usage") is True

    # -----------------------------------------------------------------------
    # Test 7: Truncation skips ingestion
    # -----------------------------------------------------------------------

    def test_truncation_skips_ingestion(self, mock_upstream):
        """max_tee_bytes truncation causes response to pass but skips ingestion."""
        url, _captured = mock_upstream
        ingested: list = []

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
            max_tee_bytes=10,  # Tiny: response will exceed this
        )
        proxy = ProxyApp(config, collector=collector)

        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Truncate me"}],
                },
                headers={"X-ClawLoop-Run-Id": "run-truncate-1"},
            )

            # Client still gets the full response
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "mock reply"

            # Deterministic: poll briefly to confirm nothing was ingested
            _assert_no_ingestion(ingested)

        assert len(ingested) == 0, "Truncated traces should not be ingested"
