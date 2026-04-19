"""Tests for the demo math CLI entry point (issue #31).

Covers:
  - clawloop demo math --dry-run  (via CLI dispatch)
  - python -m clawloop.demo_math --dry-run  (module entry point)
  - python examples/demo_math.py --dry-run  (shim, clone-based)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    *args: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
    )


def _real_llm_env() -> dict[str, str] | None:
    env = os.environ.copy()
    if env.get("CLAWLOOP_TASK_MODEL") and env.get("CLAWLOOP_REFLECTOR_MODEL"):
        return env
    if env.get("OPENAI_API_KEY"):
        env["CLAWLOOP_TASK_MODEL"] = "openai/gpt-4o-mini"
        env["CLAWLOOP_REFLECTOR_MODEL"] = "openai/gpt-5.4-nano"
        return env
    if env.get("GEMINI_API_KEY"):
        env["CLAWLOOP_TASK_MODEL"] = "gemini/gemini-2.0-flash-lite"
        env["CLAWLOOP_REFLECTOR_MODEL"] = "gemini/gemini-2.5-flash-lite"
        return env
    if env.get("ANTHROPIC_API_KEY"):
        return env
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDemoMathModule:
    """clawloop.demo_math imported directly."""

    def test_main_dry_run_succeeds(self, tmp_path: Path) -> None:
        """main(["--dry-run"]) completes without raising."""
        from clawloop.demo_math import main

        # Should not raise; if it does the test fails naturally.
        output_file = tmp_path / "playbook.json"
        main(["--dry-run", "--output", str(output_file)])

    def test_main_imports_cleanly(self) -> None:
        """The module is importable as a package-resident module."""
        import clawloop.demo_math as dm

        assert callable(dm.main)
        assert callable(dm.parse_args)


class TestDemoMathCLI:
    """clawloop demo math --dry-run via the installed CLI dispatcher."""

    def test_cli_demo_math_dry_run(self) -> None:
        result = _run(sys.executable, "-m", "clawloop.cli", "demo", "math", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "Demo complete" in result.stdout

    def test_cli_demo_math_help(self) -> None:
        result = _run(sys.executable, "-m", "clawloop.cli", "demo", "math", "--help")
        assert result.returncode == 0, result.stderr
        assert "--dry-run" in result.stdout

    def test_cli_help_shows_demo_subcommand(self) -> None:
        result = _run(sys.executable, "-m", "clawloop.cli", "--help")
        assert result.returncode == 0, result.stderr
        assert "demo" in result.stdout


class TestDemoMathModuleEntryPoint:
    """python -m clawloop.demo_math --dry-run"""

    def test_module_entry_point_dry_run(self) -> None:
        result = _run(sys.executable, "-m", "clawloop.demo_math", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "Demo complete" in result.stdout


class TestDemoMathShim:
    """examples/demo_math.py --dry-run (clone-based thin shim)."""

    def test_examples_shim_dry_run(self) -> None:
        shim = REPO_ROOT / "examples" / "demo_math.py"
        assert shim.exists(), "examples/demo_math.py shim is missing"
        result = _run(sys.executable, str(shim), "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "Demo complete" in result.stdout

    def test_examples_shim_is_thin(self) -> None:
        """The shim should not contain the full implementation — it must delegate."""
        shim = (REPO_ROOT / "examples" / "demo_math.py").read_text()
        # Real impl lives in clawloop/demo_math.py; shim just imports from there
        assert "from clawloop.demo_math import main" in shim
        # Shim must not duplicate the learning loop logic
        assert "ClawLoopAgent" not in shim
        assert "MathEnvironment" not in shim


class TestDemoMathRealLLMSmoke:
    """Real provider smoke tests for the demo entry points.

    These are skipped unless a provider API key is present.  They intentionally
    use one iteration and one episode to validate wiring without making the
    test suite depend on a long or costly benchmark run.
    """

    def test_cli_demo_math_real_llm_smoke(self, tmp_path: Path) -> None:
        env = _real_llm_env()
        if env is None:
            pytest.skip("No real LLM API key configured")
        output_file = tmp_path / "cli-playbook.json"
        result = _run(
            sys.executable,
            "-m",
            "clawloop.cli",
            "demo",
            "math",
            "--iterations",
            "1",
            "--episodes",
            "1",
            "--output",
            str(output_file),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "Demo complete" in result.stdout

    def test_module_entry_point_real_llm_smoke(self, tmp_path: Path) -> None:
        env = _real_llm_env()
        if env is None:
            pytest.skip("No real LLM API key configured")
        output_file = tmp_path / "module-playbook.json"
        result = _run(
            sys.executable,
            "-m",
            "clawloop.demo_math",
            "--iterations",
            "1",
            "--episodes",
            "1",
            "--output",
            str(output_file),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "Demo complete" in result.stdout

    def test_examples_shim_real_llm_smoke(self, tmp_path: Path) -> None:
        env = _real_llm_env()
        if env is None:
            pytest.skip("No real LLM API key configured")
        output_file = tmp_path / "shim-playbook.json"
        shim = REPO_ROOT / "examples" / "demo_math.py"
        result = _run(
            sys.executable,
            str(shim),
            "--iterations",
            "1",
            "--episodes",
            "1",
            "--output",
            str(output_file),
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "Demo complete" in result.stdout
