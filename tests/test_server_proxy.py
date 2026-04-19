"""Tests for proxy mount integration in clawloop-server."""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig
from clawloop.server import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


MOCK_COMPLETION = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "mounted ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    "model": "mock",
}

SSE_CHUNKS = (
    b'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n'
    b"data: [DONE]\n\n"
)


async def _mock_chat(request: Request):
    body = await request.json()
    if body.get("stream"):

        async def _gen():
            for line in SSE_CHUNKS.split(b"\n\n"):
                if line:
                    yield line + b"\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")
    return JSONResponse(MOCK_COMPLETION)


@pytest.fixture(scope="module")
def mock_upstream():
    """Run mock upstream on random port, return base URL."""
    port = _find_free_port()
    app = Starlette(
        routes=[Route("/chat/completions", _mock_chat, methods=["POST"])],
    )
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
        raise RuntimeError("Mock upstream did not start in time")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=3)


def _proxy_config(upstream_url: str, **overrides) -> ProxyConfig:
    defaults = dict(
        upstream_url=upstream_url,
        upstream_api_key=SecretStr("sk-test"),
        bench_mode=True,
        max_post_process_tasks=1,
    )
    defaults.update(overrides)
    return ProxyConfig(**defaults)


# ---------------------------------------------------------------------------
# C2: Route prefix — server mount
# ---------------------------------------------------------------------------


class TestServerProxyMount:
    def test_v1_routes_not_mounted_by_default(self, tmp_path):
        """Without proxy config, /v1 should 404."""
        seed = tmp_path / "seed.txt"
        seed.write_text("You are helpful.")
        app = create_app(seed_prompt_path=str(seed))
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 404 or resp.status_code == 405

    def test_proxy_mount_serves_v1(self, mock_upstream, tmp_path):
        """With proxy_config, /v1/chat/completions should reach the proxy handler."""
        seed = tmp_path / "seed.txt"
        seed.write_text("You are helpful.")
        cfg = _proxy_config(mock_upstream)
        app = create_app(seed_prompt_path=str(seed), proxy_config=cfg)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-ClawLoop-Run-Id": "run-mount-1"},
            )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "mounted ok"


# ---------------------------------------------------------------------------
# C1: Streaming uses send(stream=True)
# ---------------------------------------------------------------------------


class TestProxyStreaming:
    def test_streaming_returns_sse_chunks(self, mock_upstream):
        """Streaming request should yield SSE chunks without buffering them."""
        cfg = _proxy_config(mock_upstream)
        proxy = ProxyApp(config=cfg)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                headers={"X-ClawLoop-Run-Id": "run-stream-1"},
            )
        assert resp.status_code == 200
        body = resp.text
        assert "hello" in body
        assert "world" in body
        assert "[DONE]" in body


# ---------------------------------------------------------------------------
# I2: redaction_hook
# ---------------------------------------------------------------------------


class TestRedactionHook:
    def test_redaction_hook_is_called(self, mock_upstream):
        """redaction_hook should be applied before collector.ingest_external."""
        redacted = []

        def hook(body: dict) -> dict:
            redacted.append(True)
            # Strip messages for redaction test
            msgs = body.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    m["content"] = "[REDACTED]"
            return body

        cfg = _proxy_config(mock_upstream, redaction_hook=hook)
        proxy = ProxyApp(config=cfg)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "secret"}]},
                headers={"X-ClawLoop-Run-Id": "run-redact-1"},
            )
        assert resp.status_code == 200
        # Give worker time to process
        time.sleep(0.5)
        assert len(redacted) >= 1

    def test_redaction_hook_error_drops_item(self, mock_upstream):
        """If redaction_hook raises, the item should be dropped (not crash)."""

        def bad_hook(body: dict) -> dict:
            raise ValueError("redaction failed")

        cfg = _proxy_config(mock_upstream, redaction_hook=bad_hook)
        proxy = ProxyApp(config=cfg)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-ClawLoop-Run-Id": "run-redact-err-1"},
            )
        # The proxy should still return the upstream response successfully
        assert resp.status_code == 200
