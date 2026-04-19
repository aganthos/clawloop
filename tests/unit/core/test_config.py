"""Unit tests for clawloop.config.load_env."""

from __future__ import annotations

import importlib
import os


def _reload_module():
    import clawloop.config as m

    importlib.reload(m)
    return m


def test_load_env_is_idempotent(monkeypatch, tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text("CLAWLOOP_TEST_ONLY_KEY=hello\n")
    monkeypatch.setenv("CLAWLOOP_ENV_FILE", str(envfile))
    monkeypatch.delenv("CLAWLOOP_TEST_ONLY_KEY", raising=False)

    m = _reload_module()
    loaded_first = m.load_env()
    loaded_second = m.load_env()
    assert len(loaded_first) >= 1
    assert loaded_second == []  # idempotent
    assert os.environ["CLAWLOOP_TEST_ONLY_KEY"] == "hello"


def test_load_env_override_path_is_honored(monkeypatch, tmp_path):
    envfile = tmp_path / "custom.env"
    envfile.write_text("CLAWLOOP_TEST_OVERRIDE=1\n")
    monkeypatch.setenv("CLAWLOOP_ENV_FILE", str(envfile))
    monkeypatch.delenv("CLAWLOOP_TEST_OVERRIDE", raising=False)

    m = _reload_module()
    m.load_env(force=True)
    assert os.environ["CLAWLOOP_TEST_OVERRIDE"] == "1"


def test_load_env_never_overrides_existing(monkeypatch, tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text("CLAWLOOP_TEST_KEEP=from_env_file\n")
    monkeypatch.setenv("CLAWLOOP_ENV_FILE", str(envfile))
    monkeypatch.setenv("CLAWLOOP_TEST_KEEP", "from_process")

    m = _reload_module()
    m.load_env(force=True)
    # Existing var survives — .env never overrides process-level.
    assert os.environ["CLAWLOOP_TEST_KEEP"] == "from_process"


def test_load_env_missing_file_is_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWLOOP_ENV_FILE", str(tmp_path / "nonexistent.env"))
    m = _reload_module()
    # Must not raise even if override path does not exist.
    result = m.load_env(force=True)
    # result may be empty, but crucially no exception.
    assert isinstance(result, list)
