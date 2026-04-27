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
