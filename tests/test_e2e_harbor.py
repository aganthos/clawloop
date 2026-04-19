"""End-to-end test: Harbor trials + ClawLoop harness learning.

Test A: Adapter execution — oracle agent on hello-world, no LLM needed.
Test B: Harness learning on BFCL — terminus-2 agent on real function-calling
        tasks, reflector learns from traces.

Requires:
- Docker running locally
- Harbor submodule initialized (clawloop/benchmarks/harbor)
- For Test B: ANTHROPIC_API_KEY, GOOGLE_API_KEY, or LLM_PROXY_URL env var
- For Test B: network access to download BFCL tasks from harbor-datasets

Run with:
    pytest tests/test_e2e_harbor.py -m e2e -s --timeout=600
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import pytest

from clawloop.core.episode import Episode
from clawloop.core.loop import AgentState, learning_loop
from clawloop.learning_layers.harness import Harness

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARBOR_ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "harbor"
HELLO_WORLD_TASK = HARBOR_ROOT / "examples" / "tasks" / "hello-world"

# BFCL simple tasks to download for Test B
BFCL_TASK_SPECS = [
    {
        "name": "bfcl-live-simple-0-0-0",
        "git_url": "https://github.com/laude-institute/harbor-datasets.git",
        "git_commit_id": "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262",
        "path": "datasets/bfcl/bfcl-live-simple-0-0-0",
    },
    {
        "name": "bfcl-live-simple-1-1-0",
        "git_url": "https://github.com/laude-institute/harbor-datasets.git",
        "git_commit_id": "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262",
        "path": "datasets/bfcl/bfcl-live-simple-1-1-0",
    },
]
N_BFCL_TASKS = len(BFCL_TASK_SPECS)
N_ITERATIONS = 2
N_EPISODES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _harbor_available() -> bool:
    try:
        from harbor.trial.trial import Trial  # noqa: F401

        return True
    except ImportError:
        return False


def _proxy_available() -> bool:
    url = os.environ.get("LLM_PROXY_URL", "")
    key = os.environ.get("LLM_PROXY_KEY", "")
    if not url or not key:
        return False
    try:
        import httpx
    except ImportError:
        return False
    try:
        r = httpx.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _get_cheapest_model_config() -> tuple[str, dict]:
    """Return (litellm_model_string, extra_kwargs) for cheapest available LLM.

    Returns model config for reflector. Also used as terminus-2 model.
    Tries: proxy > Gemini flash lite > Haiku.
    """
    proxy_url = os.environ.get("LLM_PROXY_URL", "")
    proxy_key = os.environ.get("LLM_PROXY_KEY", "")
    proxy_model = os.environ.get("LLM_PROXY_MODEL", "claude-haiku-4-5-20251001")

    if proxy_url and proxy_key and _proxy_available():
        return f"openai/{proxy_model}", {"api_base": proxy_url, "api_key": proxy_key}

    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if google_key:
        return "gemini/gemini-2.0-flash-lite", {}

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        return "anthropic/claude-haiku-4-5-20251001", {}

    pytest.skip(
        "No LLM configured: set LLM_PROXY_URL+LLM_PROXY_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _check_harbor_prereqs():
    """Skip entire module if Docker or Harbor unavailable."""
    if not _docker_available():
        pytest.skip("Docker not available")
    if not _harbor_available():
        pytest.skip("Harbor not installed (submodule not initialized?)")
    if not HELLO_WORLD_TASK.exists():
        pytest.skip(f"Harbor hello-world task not found at {HELLO_WORLD_TASK}")


@pytest.fixture(scope="module")
def bfcl_task_dirs(tmp_path_factory, _check_harbor_prereqs):
    """Download BFCL simple tasks via Harbor's TaskClient. Returns list of task dirs."""
    from harbor.models.task.id import GitTaskId
    from harbor.tasks.client import TaskClient

    download_dir = tmp_path_factory.mktemp("bfcl_tasks")

    async def _download():
        client = TaskClient()
        task_ids = [
            GitTaskId(
                git_url=spec["git_url"],
                git_commit_id=spec["git_commit_id"],
                path=Path(spec["path"]),
            )
            for spec in BFCL_TASK_SPECS
        ]
        result = await client.download_tasks(task_ids=task_ids, output_dir=download_dir)
        return result.paths

    try:
        paths = asyncio.run(_download())
    except Exception as exc:
        pytest.skip(f"Failed to download BFCL tasks: {exc}")

    assert len(paths) == N_BFCL_TASKS, f"Expected {N_BFCL_TASKS} tasks, got {len(paths)}"
    for p in paths:
        assert (p / "instruction.md").exists(), f"Missing instruction.md in {p}"
    log.info("Downloaded %d BFCL tasks to %s", len(paths), download_dir)
    return paths


# ---------------------------------------------------------------------------
# Test A: Adapter execution (oracle, no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHarborAdapterExecution:
    """Prove the adapter correctly calls Harbor's Trial.create API."""

    def test_oracle_hello_world(self, _check_harbor_prereqs, tmp_path):
        from clawloop.environments.harbor import HarborTaskEnvironment

        trial_config = {
            "agent": {"name": "oracle"},
            "task": {},
            "trials_dir": str(tmp_path / "trials"),
        }

        env = HarborTaskEnvironment(
            task_dir=HELLO_WORLD_TASK,
            trial_config=trial_config,
        )

        import asyncio

        ep = asyncio.run(env.run_episode(AgentState()))

        assert isinstance(ep, Episode)
        assert ep.bench == "harbor"
        assert ep.task_id == "hello-world"
        assert ep.summary.filtered is False
        assert ep.id, "Episode must have an id"
        # Oracle follows solution, verifier should give reward=1.0
        assert (
            ep.summary.total_reward > 0
        ), f"Oracle on hello-world should succeed, got reward={ep.summary.total_reward}"

        log.info(
            "Test A passed: bench=%s task_id=%s reward=%.2f messages=%d",
            ep.bench,
            ep.task_id,
            ep.summary.total_reward,
            len(ep.messages),
        )


# ---------------------------------------------------------------------------
# Test B: Harness learning (terminus-2, real LLM)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHarborBFCLHarnessLearning:
    """Prove ClawLoop harness learning works on real BFCL function-calling tasks."""

    def test_harness_learns_from_bfcl_traces(self, bfcl_task_dirs, tmp_path):
        from clawloop.environments.harbor import HarborAdapter, HarborTaskEnvironment

        model, model_kwargs = _get_cheapest_model_config()
        log.info("Using model %s on %d BFCL tasks", model, len(bfcl_task_dirs))

        # -- Build Harbor environments for each BFCL task --
        envs = []
        for task_dir in bfcl_task_dirs:
            trial_config = {
                "agent": {
                    "name": "terminus-2",
                    "model_name": model,
                    "kwargs": {
                        "store_all_messages": True,
                        **(
                            {"api_base": model_kwargs["api_base"]}
                            if "api_base" in model_kwargs
                            else {}
                        ),
                    },
                },
                "task": {},
                "trials_dir": str(tmp_path / "trials"),
            }
            envs.append(
                HarborTaskEnvironment(
                    task_dir=task_dir,
                    trial_config=trial_config,
                )
            )

        adapter = HarborAdapter(envs=envs)
        task_ids = [env.task_id for env in envs]

        # -- Build harness with real reflector --
        from clawloop.core.reflector import Reflector
        from clawloop.harness_backends.local import LocalEvolver
        from clawloop.llm import LiteLLMClient

        reflector_client = LiteLLMClient(
            model=model,
            **({"api_base": model_kwargs.get("api_base")} if "api_base" in model_kwargs else {}),
            **({"api_key": model_kwargs.get("api_key")} if "api_key" in model_kwargs else {}),
        )
        reflector = Reflector(client=reflector_client)
        evolver = LocalEvolver(reflector=reflector)

        harness = Harness(
            system_prompts={
                "harbor": (
                    "You are a function-calling assistant. Analyze the user request, "
                    "determine the correct function and parameters, and write the "
                    "result as a JSON array to /app/result.json."
                )
            },
            evolver=evolver,
        )

        agent_state = AgentState(harness=harness)
        initial_state_hash = agent_state.state_id().combined_hash

        # -- Run learning loop --
        log.info(
            "Starting learning loop: %d iterations, %d episodes, %d tasks",
            N_ITERATIONS,
            N_EPISODES,
            len(task_ids),
        )
        agent_state, state_id = learning_loop(
            adapter=adapter,
            agent_state=agent_state,
            tasks=task_ids,
            n_episodes=N_EPISODES,
            n_iterations=N_ITERATIONS,
            active_layers=["harness"],
            output_dir=str(tmp_path / "bfcl_run"),
        )

        # -- Assertions --

        # State ID changed (learning happened)
        assert (
            state_id.combined_hash != initial_state_hash
        ), "State ID should change after learning"

        # Playbook version incremented
        assert agent_state.harness.playbook_version > 0, "Playbook version should have incremented"

        # Playbook entries are grounded in Harbor episodes
        playbook = agent_state.harness.playbook
        n_entries = len(playbook.entries)
        if n_entries > 0:
            has_sources = any(bool(entry.source_episode_ids) for entry in playbook.entries)
            assert has_sources, "At least one playbook entry should reference source episode IDs"

        log.info(
            "Test B passed: %d playbook entries, version=%d, state=%s",
            n_entries,
            agent_state.harness.playbook_version,
            state_id.combined_hash[:12],
        )
