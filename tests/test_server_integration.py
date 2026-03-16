"""Integration test: full ingest→feedback→metrics loop via HTTP."""

import pytest
from starlette.testclient import TestClient
from lfx.server import create_app


@pytest.fixture
def client(tmp_path):
    seed = tmp_path / "seed.txt"
    seed.write_text("You are a support agent.")
    app = create_app(seed_prompt_path=str(seed), bench="n8n", batch_size=2)
    with TestClient(app) as c:
        yield c


class TestFullLoop:
    def test_ingest_creates_episodes(self, client):
        state = client.get("/state").json()
        assert state["playbook_version"] == 0
        assert state["playbook_entries"] == []
        assert "support agent" in state["system_prompt"]

        for msg in ["Help me with refund", "My app crashes"]:
            resp = client.post("/ingest", json={
                "messages": [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": "I can help with that."},
                ],
                "metadata": {"conversation_id": f"conv-{msg[:5]}"},
            })
            assert resp.status_code == 200

        state = client.get("/state").json()
        assert state["metrics"]["episodes_collected"] == 2

        metrics = client.get("/metrics").json()
        assert metrics["episodes_collected"] == 2
        assert len(metrics["reward_trend"]) == 2

    def test_feedback_works(self, client):
        resp = client.post("/ingest", json={
            "messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response here"},
            ],
        })
        ep_id = resp.json()["episode_id"]
        fb = client.post("/feedback", json={"episode_id": ep_id, "score": -1.0})
        assert fb.status_code == 200

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
