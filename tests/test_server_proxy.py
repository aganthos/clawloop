"""Tests for proxy mount integration in clawloop-server."""

import pytest
from starlette.testclient import TestClient

from clawloop.server import create_app


class TestServerProxyMount:
    def test_v1_routes_not_mounted_by_default(self, tmp_path):
        """Without proxy config, /v1 should 404."""
        seed = tmp_path / "seed.txt"
        seed.write_text("You are helpful.")
        app = create_app(seed_prompt_path=str(seed))
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 404 or resp.status_code == 405
