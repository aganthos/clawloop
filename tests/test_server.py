"""Tests for clawloop-server HTTP endpoints."""

import pytest
from starlette.testclient import TestClient
from clawloop.server import create_app


@pytest.fixture
def client(tmp_path):
    seed = tmp_path / "seed_prompt.txt"
    seed.write_text("You are a support agent for Acme Corp.")
    app = create_app(seed_prompt_path=str(seed), bench="n8n")
    return TestClient(app)


@pytest.fixture
def protected_client(tmp_path):
    seed = tmp_path / "seed_prompt.txt"
    seed.write_text("You are a support agent for Acme Corp.")
    app = create_app(
        seed_prompt_path=str(seed),
        bench="n8n",
        server_api_key="test-server-key",
    )
    return TestClient(app)


class TestIngest:
    def test_valid_messages(self, client):
        resp = client.post("/ingest", json={
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data
        assert isinstance(data["playbook_version"], int)

    def test_with_metadata(self, client):
        resp = client.post("/ingest", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "metadata": {
                "conversation_id": "conv-1",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        })
        assert resp.status_code == 200

    def test_empty_messages_rejected(self, client):
        assert client.post("/ingest", json={"messages": []}).status_code == 422

    def test_missing_messages_rejected(self, client):
        assert client.post("/ingest", json={}).status_code == 422

    def test_invalid_role_rejected(self, client):
        resp = client.post("/ingest", json={
            "messages": [{"role": "invalid", "content": "test"}],
        })
        assert resp.status_code == 422


class TestFeedback:
    def test_on_existing_episode(self, client):
        ingest_resp = client.post("/ingest", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ],
        })
        episode_id = ingest_resp.json()["episode_id"]
        resp = client.post("/feedback", json={"episode_id": episode_id, "score": -1.0})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_unknown_episode(self, client):
        resp = client.post("/feedback", json={"episode_id": "nonexistent", "score": 1.0})
        assert resp.status_code == 404


class TestState:
    def test_returns_seed_prompt(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "Acme Corp" in data["system_prompt"]
        assert data["playbook_version"] == 0
        assert data["learning_status"] == "idle"
        assert data["last_error"] is None
        assert isinstance(data["playbook_entries"], list)
        assert isinstance(data["metrics"], dict)
        assert "prompt_updated_at" in data

    def test_requires_auth_when_server_key_is_set(self, protected_client):
        assert protected_client.get("/state").status_code == 401

    def test_accepts_auth_when_server_key_is_set(self, protected_client):
        resp = protected_client.get(
            "/state",
            headers={"Authorization": "Bearer test-server-key"},
        )
        assert resp.status_code == 200

    def test_rejects_malformed_auth_when_server_key_is_set(self, protected_client):
        resp = protected_client.get(
            "/state",
            headers={"Authorization": "Basic test-server-key"},
        )
        assert resp.status_code == 401

    def test_accepts_api_key_query_param(self, protected_client):
        resp = protected_client.get("/state?api_key=test-server-key")
        assert resp.status_code == 200

    def test_rejects_wrong_query_param(self, protected_client):
        resp = protected_client.get("/state?api_key=wrong-key")
        assert resp.status_code == 401


class TestReset:
    def test_clears_state(self, client):
        client.post("/ingest", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        })
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        state = client.get("/state").json()
        assert state["playbook_version"] == 0
        assert state["metrics"]["episodes_collected"] == 0


class TestMetrics:
    def test_returns_data(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "episodes_collected" in data
        assert "playbook_version" in data
        assert "learning_status" in data
        assert "reward_trend" in data
        assert isinstance(data["reward_trend"], list)


class TestEvents:
    def test_endpoint_registered(self, tmp_path):
        """Verify the /events route is registered with correct media type.

        Starlette's TestClient consumes the full streaming body before
        returning, so testing persistent SSE streams requires verifying the
        route metadata directly rather than making a live HTTP request.
        """
        from starlette.routing import Route
        from clawloop.server import events as events_handler

        seed = tmp_path / "seed.txt"
        seed.write_text("You are a support agent.")
        app = create_app(seed_prompt_path=str(seed), bench="n8n")

        sse_routes = [
            r for r in app.routes
            if isinstance(r, Route) and r.path == "/events"
        ]
        assert len(sse_routes) == 1, "/events route must be registered"
        assert "GET" in sse_routes[0].methods, "/events must accept GET"
        assert sse_routes[0].endpoint is events_handler


class TestIntegration:
    def test_ingest_and_metrics(self, client):
        for text in ["Help with refund", "App crashes"]:
            client.post("/ingest", json={
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "I can help with that."},
                ],
            })
        metrics = client.get("/metrics").json()
        assert metrics["episodes_collected"] == 2
        assert len(metrics["reward_trend"]) == 2

    def test_reset_clears_everything(self, client):
        client.post("/ingest", json={
            "messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ],
        })
        client.post("/reset")
        state = client.get("/state").json()
        assert state["playbook_version"] == 0
        assert state["metrics"]["episodes_collected"] == 0
        assert state["playbook_entries"] == []
