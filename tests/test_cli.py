"""CLI smoke tests.

`run` dispatches to ``train()``; `eval` is still disabled and emits a redirect.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "examples" / "configs"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "clawloop.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# eval (still disabled)
# ---------------------------------------------------------------------------


def test_eval_subcommand_prints_redirect_and_exits_nonzero():
    result = _run_cli("eval")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "clawloop run" in combined


def test_eval_redirect_ignores_legacy_flags_with_values():
    result = _run_cli("eval", "--bench", "entropic", "--config", "/tmp/nope.json")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "clawloop run" in combined


# ---------------------------------------------------------------------------
# demo (regression guard from PR #49)
# ---------------------------------------------------------------------------


def test_demo_math_dry_run_still_works():
    result = _run_cli("demo", "math", "--dry-run", "--iterations", "1", "--episodes", "1")
    assert result.returncode == 0, f"demo math failed: {result.stderr}"


# ---------------------------------------------------------------------------
# run: every public config validates as TrainConfig (no `Unknown env_type`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIGS_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_public_configs_validate_as_trainconfig(config_path: Path):
    """Every JSON under examples/configs/ must instantiate TrainConfig and
    name a known env_type. Acceptance criterion from #50: no `Unknown
    benchmark` failures from public configs."""
    from clawloop.train import ENV_BUILDERS, TrainConfig

    raw = json.loads(config_path.read_text())
    cfg = TrainConfig(**raw)  # raises pydantic.ValidationError on schema drift
    assert cfg.env_type in ENV_BUILDERS, (
        f"{config_path.name} uses env_type={cfg.env_type!r} "
        f"which is not in ENV_BUILDERS ({sorted(ENV_BUILDERS)})"
    )


# ---------------------------------------------------------------------------
# run: math happy path with --dry-run
# ---------------------------------------------------------------------------


def test_run_math_harness_dry_run_smoke(tmp_path: Path):
    """`clawloop run <math config> --dry-run` runs end-to-end with mocks."""
    raw = json.loads((CONFIGS_DIR / "math_harness.json").read_text())
    raw["n_iterations"] = 1
    raw["episodes_per_iter"] = 1
    cfg_path = tmp_path / "math_tiny.json"
    cfg_path.write_text(json.dumps(raw))

    result = _run_cli("run", str(cfg_path), "--dry-run")
    assert result.returncode == 0, f"run math --dry-run failed: {result.stderr}"


# ---------------------------------------------------------------------------
# run: missing config path surfaces a real error (not a redirect)
# ---------------------------------------------------------------------------


def test_run_missing_config_errors():
    result = _run_cli("run", "/tmp/clawloop-does-not-exist.json")
    assert result.returncode != 0
    # Should be a FileNotFoundError, not the old disabled-redirect text.
    combined = result.stdout + result.stderr
    assert "train_runner.py" not in combined


# ---------------------------------------------------------------------------
# --dry-run on envs that bypass _make_llm_client (PR #60 review comment 1)
# ---------------------------------------------------------------------------
# These envs (taubench, entropic, openclaw) construct or drive LLM calls
# inside their adapter rather than via train._make_llm_client. Stubbing the
# env builder under --dry-run is what makes the flag actually safe — no
# real API calls regardless of env_type or presence of API keys.


def _write_tiny(tmp_path: Path, src: Path) -> Path:
    raw = json.loads(src.read_text())
    raw["n_iterations"] = 1
    raw["episodes_per_iter"] = 1
    out = tmp_path / src.name
    out.write_text(json.dumps(raw))
    return out


def test_run_taubench_dry_run_no_api_calls(tmp_path: Path, monkeypatch):
    """Even without tau2 / API keys, --dry-run on taubench must succeed
    via the stub adapter and never touch the real ENV_BUILDER."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg_path = _write_tiny(tmp_path, CONFIGS_DIR / "taubench_harness.json")
    result = _run_cli("run", str(cfg_path), "--dry-run")
    assert result.returncode == 0, f"taubench dry-run failed: {result.stderr}"


def test_run_entropic_dry_run_no_api_calls(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg_path = _write_tiny(tmp_path, CONFIGS_DIR / "entropic_harness.json")
    result = _run_cli("run", str(cfg_path), "--dry-run")
    assert result.returncode == 0, f"entropic dry-run failed: {result.stderr}"


def test_run_openclaw_dry_run_no_api_calls(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg_path = _write_tiny(tmp_path, CONFIGS_DIR / "openclaw_proxy.json")
    result = _run_cli("run", str(cfg_path), "--dry-run")
    assert result.returncode == 0, f"openclaw dry-run failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Dry-run via dependency injection (issue #64)
# ---------------------------------------------------------------------------


def test_dry_run_overrides_map_roles_correctly():
    """`_build_dry_run_overrides(config)` returns a `(factory, builders)`
    pair. The factory maps each `LLMClientConfig` in `config.llm_clients`
    to the correct mock based on role — purely via closure, no module-level
    patching required. Regression for issue #64."""
    import clawloop.cli as _cli
    from clawloop.demo_math import MockTaskClient
    from clawloop.llm import MockLLMClient
    from clawloop.train import LLMClientConfig, TrainConfig

    cfg = TrainConfig(
        mode="harness_learning",
        env_type="math",
        llm_clients={
            "reflector": LLMClientConfig(model="anthropic/claude-sonnet-4"),
            "task": LLMClientConfig(model="anthropic/claude-haiku-4"),
        },
    )

    factory, _builders = _cli._build_dry_run_overrides(cfg)
    assert isinstance(factory(cfg.llm_clients["reflector"]), MockLLMClient)
    assert isinstance(factory(cfg.llm_clients["task"]), MockTaskClient)


def test_dry_run_overrides_stub_bypassing_env():
    """For an env_type whose builder bypasses `make_llm_client`
    (e.g. taubench), `_build_dry_run_overrides` returns an `env_builders`
    override whose builder yields a `_StubAdapter`. For an env_type that
    uses `make_llm_client` (math), it returns the global `ENV_BUILDERS`
    unchanged."""
    import clawloop.cli as _cli
    import clawloop.train as _train
    from clawloop.train import LLMClientConfig, TrainConfig

    cfg_taubench = TrainConfig(
        mode="harness_learning",
        env_type="taubench",
        llm_clients={"reflector": LLMClientConfig(model="x")},
        env_config={"domain": "retail"},
    )
    _, builders = _cli._build_dry_run_overrides(cfg_taubench)
    assert builders is not _train.ENV_BUILDERS  # override built
    adapter, tasks = builders["taubench"](cfg_taubench, cfg_taubench.llm_clients, lambda _: None)
    assert isinstance(adapter, _cli._StubAdapter)
    assert tasks  # non-empty

    cfg_math = TrainConfig(
        mode="harness_learning",
        env_type="math",
        llm_clients={
            "reflector": LLMClientConfig(model="r"),
            "task": LLMClientConfig(model="t"),
        },
    )
    _, builders_math = _cli._build_dry_run_overrides(cfg_math)
    assert builders_math is _train.ENV_BUILDERS  # no override for math


def test_dry_run_does_not_mutate_module_state(tmp_path: Path):
    """After `cmd_run --dry-run`, `_make_llm_client` and `ENV_BUILDERS` on
    `clawloop.train` are unchanged. Regression for the global-mutation
    footgun that the prior monkey-patching approach created."""
    import argparse as _argparse

    import clawloop.train as _train
    from clawloop.cli import cmd_run

    original_make_llm = _train._make_llm_client
    original_builders_dict = dict(_train.ENV_BUILDERS)

    raw = json.loads((CONFIGS_DIR / "math_harness.json").read_text())
    raw["n_iterations"] = 1
    raw["episodes_per_iter"] = 1
    cfg_path = tmp_path / "math_tiny.json"
    cfg_path.write_text(json.dumps(raw))

    args = _argparse.Namespace(config=cfg_path, dry_run=True)
    cmd_run(args)

    assert _train._make_llm_client is original_make_llm
    assert dict(_train.ENV_BUILDERS) == original_builders_dict


def test_public_llm_client_config_schema_excludes_dry_run_vocab():
    """The public `LLMClientConfig` schema must not carry any dry-run
    testing vocabulary. After issue #64 the role lives entirely in a
    closure inside `_build_dry_run_overrides` — no field, no subclass."""
    from clawloop.train import LLMClientConfig

    schema = LLMClientConfig.model_json_schema()
    properties = set(schema.get("properties", {}).keys())
    assert (
        "dry_run_role" not in properties
    ), f"LLMClientConfig should not expose dry_run_role; got {sorted(properties)}"

    dumped = LLMClientConfig(model="anthropic/x").model_dump()
    assert (
        "dry_run_role" not in dumped
    ), f"vanilla LLMClientConfig.model_dump should not include dry_run_role; got keys {sorted(dumped)}"
