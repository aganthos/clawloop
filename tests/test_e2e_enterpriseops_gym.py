"""End-to-end test: EnterpriseOps-Gym (Teams domain) + ClawLoop harness learning.

Requires:
- Docker with the Teams MCP server image pulled
- ANTHROPIC_API_KEY env var (or an LLM config file)
- Network access to HuggingFace for dataset download

Run with:
    pytest tests/test_e2e_enterpriseops_gym.py -m e2e -s --timeout=600
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from clawloop.core.episode import Episode
from clawloop.core.loop import AgentState, learning_loop
from clawloop.learning_layers.harness import Harness

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "shivakrishnareddyma225/enterpriseops-gym-mcp-teams:latest"
CONTAINER_NAME = "clawloop-e2e-eog-teams"
GYM_ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "enterpriseops-gym"
HF_DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
DOMAIN = "teams"
MODE = "oracle"
N_TASKS = 3
N_ITERATIONS = 2
N_EPISODES = 2


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _image_available(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _wait_for_server(port: int, timeout: float = 60.0) -> bool:
    """Poll until the MCP server responds on the given port."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# LLM config helper
# ---------------------------------------------------------------------------


def _proxy_available() -> bool:
    """Check if a local OpenAI-compatible proxy is running (configured via env vars)."""
    url = os.environ.get("LLM_PROXY_URL", "")
    key = os.environ.get("LLM_PROXY_KEY", "")
    if not url or not key:
        return False
    try:
        import httpx

        r = httpx.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _create_llm_config(tmp_dir: Path) -> Path:
    """Create an LLM config for the benchmark agent.

    Tries (in order):
    1. Local proxy via LLM_PROXY_URL + LLM_PROXY_KEY + LLM_PROXY_MODEL env vars
    2. Google Gemini flash lite via GOOGLE_API_KEY / GEMINI_API_KEY
    3. Anthropic Haiku via ANTHROPIC_API_KEY
    """
    proxy_url = os.environ.get("LLM_PROXY_URL", "")
    proxy_key = os.environ.get("LLM_PROXY_KEY", "")
    proxy_model = os.environ.get("LLM_PROXY_MODEL", "claude-haiku-4-5-20251001")

    if proxy_url and proxy_key and _proxy_available():
        config = {
            "llm_provider": "vllm",
            "llm_model": proxy_model,
            "llm_api_key": proxy_key,
            "llm_api_endpoint": proxy_url,
            "temperature": 0.1,
            "max_tokens": 8192,
        }
    else:
        google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if google_key:
            config = {
                "llm_provider": "google",
                "llm_model": "gemini-2.0-flash-lite",
                "llm_api_key": google_key,
                "temperature": 0.1,
                "max_tokens": 8192,
            }
        elif anthropic_key:
            config = {
                "llm_provider": "anthropic",
                "llm_model": "claude-haiku-4-5-20251001",
                "llm_api_key": anthropic_key,
                "temperature": 0.1,
                "max_tokens": 8192,
            }
        else:
            pytest.skip(
                "No LLM configured: set LLM_PROXY_URL+LLM_PROXY_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY"
            )

    config_path = tmp_dir / "llm_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def docker_teams_server():
    """Start the Teams MCP server in Docker, yield the port, stop on teardown."""
    if not _docker_available():
        pytest.skip("Docker not available")
    if not _image_available(DOCKER_IMAGE):
        pytest.skip(f"Docker image {DOCKER_IMAGE} not pulled")

    port = _find_free_port()
    log.info("Starting Teams MCP server on port %d", port)

    # Stop any leftover container from a previous run
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        timeout=10,
    )

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{port}:8005",
            DOCKER_IMAGE,
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    if not _wait_for_server(port, timeout=90):
        # Grab logs for debugging, then clean up before failing
        logs = subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, timeout=10)
        pytest.fail(
            f"Teams MCP server failed to start on port {port}.\nLogs:\n{logs.stdout}\n{logs.stderr}"
        )

    log.info("Teams MCP server ready on port %d", port)
    yield port

    # Teardown
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, timeout=10)


@pytest.fixture(scope="module")
def llm_config_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("eog_llm")
    return _create_llm_config(tmp_dir)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestEnterpriseOpsGymHarnessLearning:
    """Real e2e: Docker MCP server + real LLM + harness learning loop."""

    def test_harness_learns_from_enterprise_tasks(
        self,
        docker_teams_server,
        llm_config_path,
        tmp_path,
    ):
        from clawloop.environments.enterpriseops_gym import build_adapter_from_hf

        port = docker_teams_server

        # Patch the MCP server URL in the task configs.
        # build_adapter_from_hf downloads configs from HF — we need to
        # post-process them to point at our local Docker container.
        adapter, task_ids = build_adapter_from_hf(
            domain=DOMAIN,
            llm_config_path=llm_config_path,
            gym_root=GYM_ROOT,
            mode=MODE,
            max_tasks=N_TASKS,
        )

        # Patch MCP server URLs in all task configs to point at our container
        for env in adapter._envs.values():
            config_path = env._config_path
            with open(config_path) as f:
                config_data = json.load(f)

            # Patch single-gym URL
            if "mcp_server_url" in config_data:
                config_data["mcp_server_url"] = f"http://localhost:{port}"

            # Patch multi-gym URLs
            if "gym_servers_config" in config_data:
                for server in config_data["gym_servers_config"]:
                    server["mcp_server_url"] = f"http://localhost:{port}"

            with open(config_path, "w") as f:
                json.dump(config_data, f)

        # Build harness with a base system prompt and reflector
        from clawloop.llm import LiteLLMClient

        # Use cheapest available model for reflector
        proxy_url = os.environ.get("LLM_PROXY_URL", "")
        proxy_key = os.environ.get("LLM_PROXY_KEY", "")
        proxy_model = os.environ.get("LLM_PROXY_MODEL", "claude-haiku-4-5-20251001")

        if proxy_url and proxy_key and _proxy_available():
            reflector_client = LiteLLMClient(
                model=f"openai/{proxy_model}",
                api_base=proxy_url,
                api_key=proxy_key,
            )
        elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            reflector_client = LiteLLMClient(model="gemini/gemini-2.0-flash-lite")
        else:
            reflector_client = LiteLLMClient(model="anthropic/claude-haiku-4-5-20251001")

        harness = Harness(
            system_prompts={
                "enterpriseops-gym": (
                    "You are an enterprise operations assistant. Use the available tools "
                    "to complete tasks in the Teams application. Think step by step about "
                    "what data you need and which tools to call."
                )
            },
        )

        # Set up evolver with reflector for learning
        from clawloop.core.reflector import Reflector
        from clawloop.harness_backends.local import LocalEvolver

        reflector = Reflector(client=reflector_client)
        evolver = LocalEvolver(reflector=reflector)
        harness.evolver = evolver

        agent_state = AgentState(harness=harness)

        # Limit tasks to what we have
        tasks_to_use = task_ids[:N_TASKS]

        log.info(
            "Starting learning loop: %d tasks, %d iterations, %d episodes/iter",
            len(tasks_to_use),
            N_ITERATIONS,
            N_EPISODES,
        )

        # --- Pre-flight: verify adapter produces valid episodes ---
        preflight_episode = adapter.run_episode(tasks_to_use[0], agent_state)
        assert isinstance(
            preflight_episode, Episode
        ), f"Adapter should return Episode, got {type(preflight_episode)}"
        assert preflight_episode.bench == "enterpriseops-gym"
        assert preflight_episode.task_id, "Episode must have a task_id"

        has_messages = len(preflight_episode.messages) > 0
        is_filtered = preflight_episode.summary.filtered
        log.info(
            "Preflight episode: %d messages, filtered=%s, reward=%.3f",
            len(preflight_episode.messages),
            is_filtered,
            preflight_episode.summary.effective_reward() if not is_filtered else 0.0,
        )

        # At minimum the adapter should return a structurally valid episode
        # (even if filtered due to infra issues, the shape must be correct)
        assert preflight_episode.id, "Episode must have an id"
        if not is_filtered:
            assert has_messages, "Non-filtered episode should have messages from MCP interaction"

        # --- Run the learning loop ---
        agent_state, state_id = learning_loop(
            adapter=adapter,
            agent_state=agent_state,
            tasks=tasks_to_use,
            n_episodes=N_EPISODES,
            n_iterations=N_ITERATIONS,
            active_layers=["harness"],
            output_dir=str(tmp_path / "eog_run"),
        )

        # --- Assertions ---

        # 1. State ID changed (learning happened)
        assert (
            state_id.combined_hash != AgentState().state_id().combined_hash
        ), "State ID should change after learning — harness should have been modified"

        # 2. Playbook version incremented (forward_backward + optim_step ran)
        assert agent_state.harness.playbook_version > 0, (
            "Playbook version should have incremented — "
            "forward_backward + optim_step should have run"
        )

        # 3. Log results for manual inspection
        playbook = agent_state.harness.playbook
        n_entries = len(playbook.entries)
        log.info(
            "E2E test passed: %d playbook entries, version=%d, state_id=%s",
            n_entries,
            agent_state.harness.playbook_version,
            state_id.combined_hash[:12],
        )
