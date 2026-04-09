"""EnterpriseOps-Gym environment adapter — runs enterprise benchmark tasks, produces ClawLoop Episodes."""
from __future__ import annotations

import atexit
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reward import RewardSignal
from clawloop.core.types import SampleContext
from clawloop.utils.async_bridge import run_async

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_gym_on_path(gym_root: Path) -> None:
    """Add the EnterpriseOps-Gym repo root to sys.path so its modules resolve.

    The gym is not an installable package, so we add its root to sys.path.
    Uses append (not insert) to minimize shadowing risk from generic module
    names like 'evaluate' and 'orchestrators'.
    """
    root_str = str(gym_root)
    if root_str not in sys.path:
        sys.path.append(root_str)


def _conversation_flow_to_messages(flow: list[dict]) -> list[Message]:
    """Convert EnterpriseOps-Gym conversation_flow dicts into ClawLoop Messages."""
    msgs: list[Message] = []
    for entry in flow:
        entry_type = entry.get("type", "")
        if entry_type == "system_message":
            msgs.append(Message(role="system", content=entry.get("content", "")))
        elif entry_type == "user_message":
            msgs.append(Message(role="user", content=entry.get("content", "")))
        elif entry_type == "ai_message":
            msgs.append(Message(role="assistant", content=entry.get("content", "")))
        elif entry_type == "tool_result":
            result_data = entry.get("result", {})
            content = json.dumps(result_data.get("result", {})) if isinstance(result_data, dict) else str(result_data)
            msgs.append(Message(role="tool", content=content, name=entry.get("tool_name", "")))
    return msgs


def _compute_step_boundaries(messages: list[Message]) -> list[int]:
    boundaries: list[int] = []
    for i, msg in enumerate(messages):
        if msg.role == "user" and (i == 0 or messages[i - 1].role != "user"):
            boundaries.append(i)
    if not boundaries and messages:
        boundaries = [0]
    return boundaries


def _build_steps(step_boundaries: list[int], reward: float) -> list[StepMeta]:
    if not step_boundaries:
        return []
    steps = []
    for i in range(len(step_boundaries)):
        is_terminal = i == len(step_boundaries) - 1
        steps.append(StepMeta(t=i, reward=reward if is_terminal else 0.0,
                              done=is_terminal, timing_ms=0.0))
    return steps


# ---------------------------------------------------------------------------
# Single-task environment
# ---------------------------------------------------------------------------

class EnterpriseOpsGymEnvironment:
    """Wraps a single EnterpriseOps-Gym task config and runs it via BenchmarkExecutor."""

    def __init__(
        self,
        config_path: Path,
        llm_config_path: Path,
        gym_root: Path,
        *,
        orchestrator: str = "react",
    ):
        self._config_path = Path(config_path)
        self._llm_config_path = Path(llm_config_path)
        self._gym_root = Path(gym_root)
        self._orchestrator = orchestrator
        _ensure_gym_on_path(self._gym_root)

    @property
    def task_id(self) -> str:
        return self._config_path.stem

    async def run_episode(self, agent_state: "AgentState") -> Episode:
        _ensure_gym_on_path(self._gym_root)
        from benchmark.executor import BenchmarkExecutor
        from evaluate import load_config
        from benchmark_utils import load_llm_configs
        from orchestrators.react import ReactOrchestrator
        from orchestrators.planner_react import PlannerReactOrchestrator
        from orchestrators.decomposing_planner import DecomposingPlannerOrchestrator

        ORCHESTRATOR_MAP = {
            "react": ReactOrchestrator,
            "planner_react": PlannerReactOrchestrator,
            "decomposing": DecomposingPlannerOrchestrator,
        }

        try:
            config = load_config(str(self._config_path))
        except Exception as e:
            log.error("Failed to load task config %s: %s", self._config_path, e)
            return self._build_episode(agent_state, filtered=True,
                                       metadata={"error": "config_load_failed", "detail": str(e)})

        # --- Inject harness system prompt ---
        if hasattr(agent_state, "harness") and agent_state.harness:
            try:
                sample_result = agent_state.harness.sample(
                    SampleContext(bench=self.task_id))
                prompt = sample_result.result().output
                if not prompt:
                    sample_result = agent_state.harness.sample(
                        SampleContext(bench="enterpriseops-gym"))
                    prompt = sample_result.result().output
                if prompt:
                    config.system_prompt = prompt
            except Exception:
                log.debug("Failed to sample system prompt from harness", exc_info=True)

        # Force single run per episode (state isolation)
        config.number_of_runs = 1

        try:
            llm_configs = load_llm_configs(str(self._llm_config_path))
            llm_config = llm_configs[0]
        except Exception as e:
            log.error("Failed to load LLM config %s: %s", self._llm_config_path, e)
            return self._build_episode(agent_state, filtered=True,
                                       metadata={"error": "llm_config_failed", "detail": str(e)})

        orchestrator_class = ORCHESTRATOR_MAP.get(self._orchestrator, ReactOrchestrator)

        executor = BenchmarkExecutor(
            config,
            llm_config=llm_config,
            orchestrator_class=orchestrator_class,
            config_path=str(self._config_path),
        )

        try:
            result = await executor.execute_benchmark()
        except Exception as e:
            log.error("Executor failed for task %s: %s", self.task_id, e)
            return self._build_episode(agent_state, filtered=True,
                                       metadata={"error": "executor_failed", "detail": str(e)})

        # Extract the first (and only) run result
        runs = result.get("runs", [])
        if not runs:
            return self._build_episode(agent_state, filtered=True,
                                       metadata={"error": "no_runs_returned"})

        run = runs[0]

        # Infra error → filtered
        if run.get("error"):
            return self._build_episode(agent_state, filtered=True,
                                       metadata={"error": "run_error", "detail": run["error"]})

        # Build Episode from conversation flow and verification results
        conversation_flow = run.get("conversation_flow", [])
        verification = run.get("verification_summary", {})
        pass_rate = verification.get("pass_rate", 0.0)

        # Map pass_rate [0, 1] → reward [-1, 1]
        reward = (pass_rate * 2.0) - 1.0

        messages = _conversation_flow_to_messages(conversation_flow)
        step_boundaries = _compute_step_boundaries(messages)
        steps = _build_steps(step_boundaries, reward)

        summary = EpisodeSummary(
            score_breakdown=run.get("verification_results"),
        )
        summary.signals["outcome"] = RewardSignal(
            name="outcome", value=reward, confidence=1.0,
        )

        metadata: dict[str, Any] = {
            "pass_rate": pass_rate,
            "overall_success": run.get("overall_success", False),
            "tools_used": run.get("tools_used", []),
            "execution_time_ms": run.get("execution_time_ms", 0),
            "verification_summary": verification,
        }

        state_id = ""
        if hasattr(agent_state, "state_id") and callable(agent_state.state_id):
            try:
                state_id = agent_state.state_id().combined_hash
            except Exception:
                log.debug("Failed to compute state_id", exc_info=True)

        return Episode(
            id=uuid4().hex,
            state_id=state_id or "",
            task_id=self.task_id,
            bench="enterpriseops-gym",
            messages=messages,
            step_boundaries=step_boundaries,
            steps=steps,
            summary=summary,
            metadata=metadata,
        )

    def _build_episode(self, agent_state: "AgentState", *, filtered: bool = False,
                       reward: float = 0.0, metadata: dict | None = None) -> Episode:
        summary = EpisodeSummary(filtered=filtered)
        if not filtered:
            summary.signals["outcome"] = RewardSignal(
                name="outcome", value=reward, confidence=1.0,
            )
        state_id = ""
        if hasattr(agent_state, "state_id") and callable(agent_state.state_id):
            try:
                state_id = agent_state.state_id().combined_hash
            except Exception:
                pass
        return Episode(
            id=uuid4().hex, state_id=state_id or "", task_id=self.task_id,
            bench="enterpriseops-gym", messages=[], step_boundaries=[],
            steps=[], summary=summary, metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Adapter (sync wrapper, implements AdapterLike)
# ---------------------------------------------------------------------------

class EnterpriseOpsGymAdapter:
    """Sync adapter for EnterpriseOps-Gym. Implements AdapterLike for learning_loop.

    Wraps a set of task configs and runs them sequentially via BenchmarkExecutor.
    Each run creates and cleans up its own database (state isolation is handled
    by the benchmark's create_database_from_file / delete_database lifecycle).

    License note: EnterpriseOps-Gym is CC BY-NC 4.0 (non-commercial use only).
    """

    def __init__(self, envs: list[EnterpriseOpsGymEnvironment]):
        self._envs: dict[str, EnterpriseOpsGymEnvironment] = {}
        for env in envs:
            if env.task_id in self._envs:
                raise ValueError(f"Duplicate task_id {env.task_id!r}")
            self._envs[env.task_id] = env

    @property
    def task_ids(self) -> list[str]:
        return list(self._envs.keys())

    def run_episode(self, task: str, agent_state: "AgentState") -> Episode:
        return run_async(self._envs[task].run_episode(agent_state))

    def run_batch(self, agent_state: "AgentState", tasks: list[str],
                  n_per_task: int = 1) -> list[Episode]:
        # Sequential execution — MCP servers are stateful, parallel runs
        # against the same domain risk state contamination.
        episodes: list[Episode] = []
        for task in tasks:
            for _ in range(n_per_task):
                episodes.append(self.run_episode(task, agent_state))
        return episodes


# ---------------------------------------------------------------------------
# Factory: build adapter from HuggingFace dataset
# ---------------------------------------------------------------------------

def build_adapter_from_hf(
    domain: str,
    llm_config_path: str | Path,
    gym_root: str | Path,
    *,
    mode: str = "oracle",
    hf_dataset: str = "ServiceNow-AI/EnterpriseOps-Gym",
    orchestrator: str = "react",
    max_tasks: int | None = None,
) -> tuple[EnterpriseOpsGymAdapter, list[str]]:
    """Download task configs from HuggingFace and build an adapter.

    Returns (adapter, task_ids) ready for learning_loop().
    """
    from datasets import load_dataset as hf_load_dataset

    _ensure_gym_on_path(Path(gym_root))

    tmp_dir = tempfile.mkdtemp(prefix="clawloop_eog_")
    atexit.register(shutil.rmtree, tmp_dir, True)
    json_string_fields = {"gym_servers_config", "verifiers"}
    hf_only_fields = {"task_id", "domain"}

    log.info("Loading EnterpriseOps-Gym tasks: dataset=%s mode=%s domain=%s",
             hf_dataset, mode, domain)
    hf_ds = hf_load_dataset(hf_dataset, mode, split=domain)

    envs: list[EnterpriseOpsGymEnvironment] = []
    for i, row in enumerate(hf_ds):
        if max_tasks is not None and i >= max_tasks:
            break
        task_id = row.get("task_id", f"task_{i}")
        file_name = f"{mode}__{domain}__{task_id}.json"
        task_dict = {}
        for k, v in row.items():
            if k in hf_only_fields:
                continue
            if k in json_string_fields and isinstance(v, str):
                v = json.loads(v)
            task_dict[k] = v
        config_path = Path(tmp_dir) / file_name
        with open(config_path, "w") as f:
            json.dump(task_dict, f)
        envs.append(EnterpriseOpsGymEnvironment(
            config_path=config_path,
            llm_config_path=Path(llm_config_path),
            gym_root=Path(gym_root),
            orchestrator=orchestrator,
        ))

    log.info("Built %d task environments in %s", len(envs), tmp_dir)
    adapter = EnterpriseOpsGymAdapter(envs)
    return adapter, adapter.task_ids
