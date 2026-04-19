"""CLI smoke tests — ensure disabled subcommands emit a truthful redirect."""

from __future__ import annotations

import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "clawloop.cli", *args],
        capture_output=True,
        text=True,
    )


def test_run_subcommand_prints_redirect_and_exits_nonzero():
    result = _run_cli("run")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "train_runner.py" in combined
    assert "clawloop demo" in combined


def test_run_subcommand_redirects_even_with_legacy_flags():
    result = _run_cli("run", "--bench", "entropic", "--iterations", "1")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "train_runner.py" in combined


def test_run_subcommand_redirects_with_global_verbose_flag():
    result = _run_cli("-v", "run", "--bench", "entropic")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "train_runner.py" in combined


def test_run_subcommand_redirects_on_help_flag():
    result = _run_cli("run", "--help")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "train_runner.py" in combined


def test_eval_subcommand_prints_redirect_and_exits_nonzero():
    result = _run_cli("eval")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "train_runner.py" in combined


def test_run_redirect_ignores_unknown_subparser_flags_with_values():
    # Regression guard for the class of failure flagged in review: a flag that
    # *takes a value* must not cause the disabled redirect to miss. Since only
    # the outer parser has globals today, we prove the intercept is robust by
    # passing a value-taking flag to the run subparser.
    result = _run_cli("run", "--config", "/tmp/does-not-exist.json")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "train_runner.py" in combined


def test_demo_math_dry_run_still_works():
    result = _run_cli("demo", "math", "--dry-run", "--iterations", "1", "--episodes", "1")
    assert result.returncode == 0, f"demo math failed: {result.stderr}"
