"""Tests for clawloop.proxy — ProxyApp with real mock upstream server."""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig
from clawloop.proxy_skills import inject_skills, strip_skills

# ---------------------------------------------------------------------------
# Fixtures — mock upstream server
# ---------------------------------------------------------------------------

MOCK_COMPLETION = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Global to capture what the mock upstream received
_last_upstream_request: dict[str, Any] = {}


async def _mock_chat_completions(request: Request) -> JSONResponse:
    """Mock upstream /chat/completions endpoint."""
    body = await request.json()
    _last_upstream_request["body"] = body
    _last_upstream_request["headers"] = dict(request.headers)
    return JSONResponse(MOCK_COMPLETION)


_mock_upstream_app = Starlette(
    routes=[Route("/chat/completions", _mock_chat_completions, methods=["POST"])],
)


@pytest.fixture(scope="module")
def mock_upstream():
    """Run mock upstream on random port, return URL."""
    port = _find_free_port()
    config = uvicorn.Config(
        _mock_upstream_app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for readiness
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


def _make_proxy_config(
    upstream_url: str,
    *,
    bench_mode: bool = True,
    proxy_key: str = "",
) -> ProxyConfig:
    return ProxyConfig(
        upstream_url=upstream_url,
        upstream_api_key=SecretStr("sk-upstream-test"),
        bench_mode=bench_mode,
        proxy_key=proxy_key,
        max_post_process_tasks=2,
    )


def _chat_body(stream: bool = False) -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# TestProxyNonStreaming
# ---------------------------------------------------------------------------


class TestProxyNonStreaming:
    def test_forwards_and_returns_response(self, mock_upstream: str) -> None:
        """Send request with Run-Id, verify 200 + correct response body."""
        config = _make_proxy_config(mock_upstream)
        proxy = ProxyApp(config=config)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body(),
                headers={"X-ClawLoop-Run-Id": "run-001"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        assert body["usage"]["total_tokens"] == 18

    def test_bench_mode_rejects_without_run_id(self, mock_upstream: str) -> None:
        """No Run-Id header in bench_mode -> 400."""
        config = _make_proxy_config(mock_upstream)
        proxy = ProxyApp(config=config)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body(),
            )
        assert resp.status_code == 400
        assert "Run-Id" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# TestProxyAuth
# ---------------------------------------------------------------------------


class TestProxyAuth:
    def test_live_mode_rejects_without_auth(self, mock_upstream: str) -> None:
        """bench_mode=False, no Authorization -> 401."""
        config = _make_proxy_config(
            mock_upstream, bench_mode=False, proxy_key="secret-key"
        )
        proxy = ProxyApp(config=config)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body(),
            )
        assert resp.status_code == 401

    def test_live_mode_accepts_with_auth(self, mock_upstream: str) -> None:
        """bench_mode=False, correct Bearer token -> 200."""
        config = _make_proxy_config(
            mock_upstream, bench_mode=False, proxy_key="secret-key"
        )
        proxy = ProxyApp(config=config)
        with TestClient(proxy.asgi_app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body(),
                headers={
                    "Authorization": "Bearer secret-key",
                    "X-ClawLoop-Session-Id": "sess-1",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# TestSkillInjection (unit-level, no upstream needed)
# ---------------------------------------------------------------------------


class TestSkillInjection:
    def test_inject_skills_prepends_system(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        result = inject_skills(msgs, "## Skills\n- do stuff")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Skills" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_strip_skills_removes_injected(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        injected = inject_skills(msgs, "## Skills\n- do stuff")
        stripped = strip_skills(injected)
        assert len(stripped) == 1
        assert stripped[0]["role"] == "user"

    def test_inject_empty_noop(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        result = inject_skills(msgs, "")
        assert result is msgs

    def test_inject_idempotent(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        once = inject_skills(msgs, "skills text")
        twice = inject_skills(once, "skills text")
        assert len(twice) == 2  # still 2, not 3


# ---------------------------------------------------------------------------
# TestPostProcessing
# ---------------------------------------------------------------------------


class TestPostProcessing:
    def test_drops_total_starts_at_zero(self, mock_upstream: str) -> None:
        config = _make_proxy_config(mock_upstream)
        proxy = ProxyApp(config=config)
        assert proxy.drops_total == 0

    def test_upstream_receives_auth_header(self, mock_upstream: str) -> None:
        """Verify the upstream receives Bearer with upstream_api_key."""
        config = _make_proxy_config(mock_upstream)
        proxy = ProxyApp(config=config)
        with TestClient(proxy.asgi_app) as client:
            _last_upstream_request.clear()
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body(),
                headers={"X-ClawLoop-Run-Id": "run-auth-test"},
            )
        assert resp.status_code == 200
        # The upstream should have seen the upstream_api_key
        assert "authorization" in _last_upstream_request.get("headers", {})
        assert (
            _last_upstream_request["headers"]["authorization"]
            == "Bearer sk-upstream-test"
        )
