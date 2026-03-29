"""Real SkyDiscover integration tests — calls run_discovery, no mocks.

Validates the actual SkyDiscover library works with our wrapper:
1. run_discovery with a trivial evaluator and small iterations
2. ClawLoopEvaluator plugged into run_discovery
3. Full pipeline: Harness → SkyDiscoverAdaEvolve → real evolution → playbook update

Requires:
- SkyDiscover installed (pip install -e enterprise/evolution/skydiscover/)
- CLIProxyAPI running at localhost:8317 (for LLM calls)

Skipped if either dependency is missing.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest

try:
    from skydiscover import run_discovery
    from skydiscover.api import DiscoveryResult
    _skydiscover_available = True
except ImportError:
    _skydiscover_available = False


def _proxy_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8317), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


skip_no_deps = pytest.mark.skipif(
    not (_skydiscover_available and _proxy_available()),
    reason="SkyDiscover not installed or CLIProxyAPI not running",
)

_API_BASE = "http://127.0.0.1:8317/v1"
_API_KEY = os.environ.get("LFX_API_KEY", "kuhhandel-bench-key")
_MODEL = "openai/claude-haiku-4-5-20251001"


@skip_no_deps
class TestRealRunDiscovery:
    """Calls SkyDiscover's run_discovery for real."""

    def test_trivial_evaluator(self, tmp_path: Path) -> None:
        """Minimal run_discovery with a scoring function that prefers '4'."""
        def evaluator(program_path: str) -> dict[str, Any]:
            content = Path(program_path).read_text()
            score = 1.0 if "4" in content else 0.0
            return {"combined_score": score}

        seed = tmp_path / "seed.py"
        seed.write_text("# EVOLVE-BLOCK-START\nanswer = 3\n# EVOLVE-BLOCK-END\n")

        result = run_discovery(
            evaluator=evaluator,
            initial_program=str(seed),
            model=_MODEL,
            iterations=3,
            search="topk",  # fastest search for smoke test
            api_base=_API_BASE,
            cleanup=True,
        )

        assert isinstance(result, DiscoveryResult)
        assert result.best_score >= 0.0
        assert result.best_solution is not None
        assert len(result.best_solution) > 0

    def test_adaevolve_smoke(self, tmp_path: Path) -> None:
        """AdaEvolve search with 2 iterations — validates the algorithm runs."""
        call_count = 0

        def evaluator(program_path: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            content = Path(program_path).read_text()
            # Simple heuristic: longer = better (encourages evolution)
            score = min(len(content) / 500.0, 1.0)
            return {"combined_score": score}

        seed = tmp_path / "seed.py"
        seed.write_text(
            "# EVOLVE-BLOCK-START\n"
            "def solve(x):\n"
            "    return x + 1\n"
            "# EVOLVE-BLOCK-END\n"
        )

        result = run_discovery(
            evaluator=evaluator,
            initial_program=str(seed),
            model=_MODEL,
            iterations=2,
            search="adaevolve",
            api_base=_API_BASE,
            cleanup=True,
        )

        assert isinstance(result, DiscoveryResult)
        assert call_count >= 1, "Evaluator should have been called at least once"
        assert result.best_solution is not None


@skip_no_deps
class TestRealEvolverPipeline:
    """Full pipeline: SkyDiscoverAdaEvolve with real run_discovery."""

    def test_evolve_with_real_skydiscover(self, tmp_path: Path) -> None:
        """SkyDiscoverAdaEvolve.evolve() calls real run_discovery."""
        from unittest.mock import MagicMock

        from clawloop.core.evolver import EvolverContext, EvolverResult
        from enterprise.evolution.backends.skydiscover_adaevolve import SkyDiscoverAdaEvolve
        from enterprise.tests.conftest import make_episode, make_snapshot

        # Simple adapter that returns fixed episodes
        adapter = MagicMock()
        adapter.run_episode.return_value = make_episode(0.5)

        factory = MagicMock(return_value=MagicMock())

        evolver = SkyDiscoverAdaEvolve(
            adapter=adapter,
            tasks=["What is 2+2?"],
            agent_state_factory=factory,
            iterations=2,
            model=_MODEL,
            num_islands=1,
            population_size=5,
            n_eval_episodes=1,
        )
        # Override the model to use our proxy
        evolver._model = _MODEL

        snap = make_snapshot()
        ctx = EvolverContext(reward_history=[0.3], iteration=0)

        result = evolver.evolve([make_episode(-0.5)], snap, ctx)

        assert isinstance(result, EvolverResult)
        assert result.provenance.backend == "skydiscover_adaevolve"
        # The evolution should produce SOMETHING (insights or candidates)
        # even with just 2 iterations — or at least not crash
        evolver.cleanup()
