"""Tests for clawloop.proxy_config — ProxyConfig validation."""
from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from clawloop.proxy_config import ProxyConfig


# ---------------------------------------------------------------------------
# Minimal valid config with defaults
# ---------------------------------------------------------------------------

class TestMinimalConfig:
    def test_minimal_valid_config(self):
        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test-key"),
        )
        assert cfg.upstream_url == "https://api.openai.com/v1"
        assert cfg.upstream_api_key.get_secret_value() == "sk-test-key"
        assert cfg.bench == "openclaw"
        assert cfg.bench_mode is True
        assert cfg.proxy_key == ""
        assert cfg.max_tee_bytes == 524288
        assert cfg.live_idle_timeout_s == 300
        assert cfg.upstream_connect_timeout_s == 10.0
        assert cfg.upstream_read_timeout_s == 120.0
        assert cfg.upstream_supports_stream_usage is True
        assert cfg.max_post_process_tasks == 8
        assert cfg.redaction_hook is None

    def test_forward_headers_constant(self):
        assert ProxyConfig.FORWARD_HEADERS == frozenset(
            {"content-type", "accept", "user-agent"}
        )


# ---------------------------------------------------------------------------
# upstream_url validation: must be https for remote hosts
# ---------------------------------------------------------------------------

class TestUpstreamUrlValidation:
    def test_rejects_http_remote(self):
        with pytest.raises(ValidationError, match="https"):
            ProxyConfig(
                upstream_url="http://remote-host.com",
                upstream_api_key=SecretStr("sk-test"),
            )

    def test_allows_https_remote(self):
        cfg = ProxyConfig(
            upstream_url="https://remote-host.com/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.upstream_url == "https://remote-host.com/v1"

    def test_allows_http_localhost(self):
        cfg = ProxyConfig(
            upstream_url="http://localhost:8080/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.upstream_url == "http://localhost:8080/v1"

    def test_allows_http_127_0_0_1(self):
        cfg = ProxyConfig(
            upstream_url="http://127.0.0.1:8080/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.upstream_url == "http://127.0.0.1:8080/v1"

    def test_allows_http_ipv6_loopback(self):
        cfg = ProxyConfig(
            upstream_url="http://[::1]:8080/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.upstream_url == "http://[::1]:8080/v1"


# ---------------------------------------------------------------------------
# bench_mode flag
# ---------------------------------------------------------------------------

class TestBenchMode:
    def test_bench_mode_default_true(self):
        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.bench_mode is True

    def test_bench_mode_explicit_false(self):
        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
            bench_mode=False,
            proxy_key="live-key-123",
        )
        assert cfg.bench_mode is False


# ---------------------------------------------------------------------------
# Live mode requires proxy_key
# ---------------------------------------------------------------------------

class TestLiveModeValidation:
    def test_live_mode_requires_proxy_key(self):
        with pytest.raises(ValidationError, match="proxy_key"):
            ProxyConfig(
                upstream_url="https://api.openai.com/v1",
                upstream_api_key=SecretStr("sk-test"),
                bench_mode=False,
                proxy_key="",
            )

    def test_live_mode_missing_proxy_key_default(self):
        with pytest.raises(ValidationError, match="proxy_key"):
            ProxyConfig(
                upstream_url="https://api.openai.com/v1",
                upstream_api_key=SecretStr("sk-test"),
                bench_mode=False,
            )

    def test_live_mode_with_proxy_key_ok(self):
        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
            bench_mode=False,
            proxy_key="my-live-key",
        )
        assert cfg.proxy_key == "my-live-key"
        assert cfg.bench_mode is False

    def test_bench_mode_no_proxy_key_ok(self):
        """In bench mode, proxy_key can be empty."""
        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
            bench_mode=True,
            proxy_key="",
        )
        assert cfg.proxy_key == ""


# ---------------------------------------------------------------------------
# Redaction hook (callable field)
# ---------------------------------------------------------------------------

class TestRedactionHook:
    def test_accepts_callable(self):
        def my_hook(d: dict) -> dict:
            return d

        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
            redaction_hook=my_hook,
        )
        assert cfg.redaction_hook is my_hook

    def test_none_by_default(self):
        cfg = ProxyConfig(
            upstream_url="https://api.openai.com/v1",
            upstream_api_key=SecretStr("sk-test"),
        )
        assert cfg.redaction_hook is None
