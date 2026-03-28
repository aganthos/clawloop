"""End-to-end integration test: mock upstream -> proxy -> EpisodeCollector."""
from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig

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

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"),
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to accept connections
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("Mock upstream did not start in time")

    yield f"http://127.0.0.1:{port}", captured

    server.should_exit = True
    thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_proxy_captures_and_ingests(self, mock_upstream):
        """Full round-trip: client -> proxy -> upstream -> collector ingest."""
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
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if len(ingested) >= 1:
                    break
                time.sleep(0.1)

        # Collector should have ingested at least one episode
        assert len(ingested) >= 1

        ep = ingested[0]
        assert ep.bench == "test-bench"
        # The episode should contain the user message and the assistant reply
        roles = [m.role for m in ep.messages]
        assert "user" in roles
        assert "assistant" in roles
