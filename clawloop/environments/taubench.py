"""TauBench (tau3) adapter — integrates sierra-research/tau2-bench (dev/tau3 branch)
into the ClawLoop learning loop.

Supports retail and airline domains (and any other tau2-registered domain).
tau2 is used as a Python library — no subprocess or external server required.

Install tau2:
    pip install "clawloop[taubench]"
    # or directly:
    pip install git+https://github.com/sierra-research/tau2-bench.git@dev/tau3
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.environments.base import EnvAdapter

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)

_AGENT_KEY = "clawloop_agent"

# Module-level names imported lazily — set to None here so tests can patch them
# without tau2 installed. _require_tau2() raises if they are still None at runtime.
try:
    from tau2.run import get_tasks, run_single_task
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
except ImportError:
    get_tasks = None          # type: ignore[assignment]
    run_single_task = None    # type: ignore[assignment]
    TextRunConfig = None      # type: ignore[assignment]
    EvaluationType = None     # type: ignore[assignment]


def _require_tau2() -> None:
    if get_tasks is None:
        raise ImportError(
            "tau2 is required for TauBenchAdapter. "
            'Install: pip install "clawloop[taubench]" or '
            "pip install git+https://github.com/sierra-research/tau2-bench.git@dev/tau3"
        )


class TauBenchAdapter(EnvAdapter):
    """EnvAdapter for tau-bench 3 (sierra-research/tau2-bench, dev/tau3 branch).

    Supports retail and airline domains. Set ``domain`` in the config dict to
    switch between them. Any other tau2-registered domain also works.

    Example config::

        {
            "domain": "retail",
            "llm_agent": "openai/claude-haiku-4-5-20251001",
            "llm_user": "openai/claude-haiku-4-5-20251001",
            "max_steps": 30,
            "max_concurrency": 8,
            "task_split": "test",
            "num_tasks": 10
        }
    """

    def setup(self, config: dict[str, Any]) -> None:
        """One-time initialization from config dict."""
        self._domain = config.get("domain", "retail")
        self._llm_agent = config.get("llm_agent", "openai/gpt-4o-mini")
        self._llm_user = config.get("llm_user", "openai/gpt-4o-mini")
        self._max_steps = int(config.get("max_steps", 30))
        self._max_concurrency = int(config.get("max_concurrency", 8))
        self._task_split = config.get("task_split", "test")
        self._num_tasks = config.get("num_tasks", None)
        self._config = config
        self._iteration_count = 0

    def list_tasks(self, split: str = "test") -> list[str]:
        """Return available task IDs for the configured domain and split."""
        _require_tau2()
        tasks = get_tasks(task_set_name=self._domain, task_split_name=split)
        return [t.id for t in tasks]

    def run_episode(self, task: Any, agent_state: "AgentState") -> Episode:
        """Run a single task. Delegates to run_batch."""
        episodes = self.run_batch(agent_state, [task])
        return episodes[0]

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks in parallel via ThreadPoolExecutor.

        Registers a ClawLoopAgent with the current harness prompt in tau2's
        registry before execution, so the evolved system prompt is injected
        for every task in this batch.
        """
        _require_tau2()

        # Pull current harness prompt
        harness_prompt = ""
        if hasattr(agent_state, "harness") and agent_state.harness:
            try:
                harness_prompt = agent_state.harness.system_prompt("taubench")
            except Exception:
                log.debug("Failed to get harness system prompt", exc_info=True)

        # Register custom agent with current harness instruction
        _register_clawloop_agent(harness_prompt)

        # Load tau2 Task objects for the requested IDs
        all_tasks = get_tasks(
            task_set_name=self._domain, task_split_name=self._task_split
        )
        task_map = {t.id: t for t in all_tasks}

        config = TextRunConfig(
            domain=self._domain,
            agent=_AGENT_KEY,
            user="user_simulator",
            llm_agent=self._llm_agent,
            llm_args_agent={},
            llm_user=self._llm_user,
            llm_args_user={},
            max_steps=self._max_steps,
        )

        state_id = ""
        if hasattr(agent_state, "state_id") and callable(agent_state.state_id):
            try:
                state_id = agent_state.state_id().combined_hash
            except Exception:
                log.debug("Failed to compute state_id", exc_info=True)

        str_ids = [str(tid) for tid in task_ids]

        def _run_one(task_id: str) -> Episode:
            task = task_map.get(task_id)
            if task is None:
                log.warning(
                    "Task %r not found in domain %r split %r",
                    task_id, self._domain, self._task_split,
                )
                return self._make_failed_episode(task_id, state_id, "task_not_found")
            try:
                # ALL_IGNORE_BASIS: use DB check + action checks as ground
                # truth; skip NL assertion evaluation (requires separate LLM
                # judge config). DB state verification is the primary signal.
                sim_run = run_single_task(
                    config, task,
                    evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
                )
                return self._map_to_episode(sim_run, task_id, state_id)
            except Exception as exc:
                log.error("tau2 run_single_task failed for %s: %s", task_id, exc)
                return self._make_failed_episode(task_id, state_id, type(exc).__name__)

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = [pool.submit(_run_one, tid) for tid in str_ids]
            # Preserve submission order. _run_one catches all exceptions internally
            # so f.result() always returns an Episode (never raises).
            episodes = [f.result() for f in futures]

        self._iteration_count += 1
        return episodes

    def _map_to_episode(
        self, sim_run: Any, task_id: str, state_id: str
    ) -> Episode:
        """Convert a tau2 SimulationRun to a ClawLoop Episode."""
        # Convert tau2 messages to ClawLoop Messages
        messages: list[Message] = []
        for m in sim_run.messages or []:
            role = getattr(m, "role", "user")
            role_str = role.value if hasattr(role, "value") else str(role)
            content = getattr(m, "content", "")
            if not isinstance(content, str):
                content = ""  # tool messages; flatten to empty string
            messages.append(Message(role=role_str, content=content))

        step_boundaries = _compute_step_boundaries(messages)

        reward_info = sim_run.reward_info
        summary = EpisodeSummary()

        if reward_info is not None:
            summary.total_reward = float(reward_info.reward)
            breakdown: dict[str, Any] = {"reward": reward_info.reward}
            if reward_info.db_check is not None:
                breakdown["db_check"] = reward_info.db_check.model_dump()
            if reward_info.env_assertions:
                breakdown["env_assertions"] = [
                    a.model_dump() for a in reward_info.env_assertions
                ]
            if reward_info.action_checks:
                breakdown["action_checks"] = [
                    a.model_dump() for a in reward_info.action_checks
                ]
            summary.score_breakdown = breakdown
        else:
            summary.total_reward = 0.0

        term = getattr(sim_run, "termination_reason", None)
        term_str = term.value if hasattr(term, "value") else str(term) if term else ""
        # MAX_ERRORS_REACHED means the episode is invalid — filter it from training
        summary.filtered = term_str == "MAX_ERRORS_REACHED"

        duration_ms = getattr(sim_run, "duration", 0.0) * 1000
        reward_val = float(reward_info.reward) if reward_info is not None else 0.0
        steps = [StepMeta(t=0, reward=reward_val, done=True, timing_ms=duration_ms)]

        return Episode(
            id=uuid4().hex,
            state_id=state_id,
            task_id=f"taubench:{task_id}",
            bench="taubench",
            messages=messages,
            step_boundaries=step_boundaries,
            steps=steps,
            summary=summary,
            metadata={
                "domain": self._domain,
                "termination_reason": term_str,
                "agent_cost": getattr(sim_run, "agent_cost", None),
                "user_cost": getattr(sim_run, "user_cost", None),
                "truncated": term_str == "MAX_STEPS_REACHED",
            },
        )

    def _make_failed_episode(
        self, task_id: str, state_id: str, reason: str
    ) -> Episode:
        """Return a negative-reward episode for tasks that could not be run (missing task, exception, etc.).

        The episode is kept unfiltered so the agent receives a -1.0 outcome signal as a
        training gradient. Only structural failures (MAX_ERRORS_REACHED) are filtered via
        summary.filtered in _map_to_episode.
        """
        from clawloop.core.reward import RewardSignal

        summary = EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=-1.0, confidence=0.5)}
        )
        return Episode(
            id=uuid4().hex,
            state_id=state_id,
            task_id=f"taubench:{task_id}",
            bench="taubench",
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=summary,
            metadata={"error": reason, "domain": self._domain},
        )

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "taubench", "episode_id": episode.id, "domain": self._domain}


# ---------------------------------------------------------------------------
# ClawLoop harness injection
# ---------------------------------------------------------------------------

# Default instruction used when no harness prompt is configured.
_DEFAULT_HARNESS_INSTRUCTION: str = (
    "You are a customer service agent. Help the user according to the policy. "
    "Be accurate, efficient, and polite."
)

# Module-level mutable cell: updated before each batch so _ClawLoopAgent
# reads the latest harness prompt at property-access time, without needing
# a new class definition per iteration.
# Thread-safety note: ClawLoop's learning loop calls run_batch() sequentially
# (one batch per iteration). Concurrent run_batch() calls with different harness
# prompts are NOT supported — they would race on this cell.
_current_harness_instruction: str = _DEFAULT_HARNESS_INSTRUCTION

# Lazily-created agent class and factory (defined once, reused every iteration).
_clawloop_agent_class: type | None = None
_clawloop_factory: Any = None


def _register_clawloop_agent(harness_instruction: str) -> None:
    """Register a ClawLoopAgent factory in tau2's registry.

    Updates the module-level prompt cell and (lazily) defines the agent class
    once. Re-registration across iterations just overwrites the registry slot —
    no new class is created per call.
    """
    global _current_harness_instruction, _clawloop_agent_class, _clawloop_factory
    # Use explicit default rather than "or" so an empty string clears the stale
    # prompt instead of leaking the previous batch's instruction.
    _current_harness_instruction = harness_instruction or _DEFAULT_HARNESS_INSTRUCTION

    if _clawloop_agent_class is None:
        from tau2.agent.llm_agent import LLMAgent, SYSTEM_PROMPT

        class _ClawLoopAgent(LLMAgent):
            @property
            def system_prompt(self) -> str:  # type: ignore[override]
                return SYSTEM_PROMPT.format(
                    agent_instruction=_current_harness_instruction,
                    domain_policy=self.domain_policy,
                )

        def _factory(
            tools: Any,
            domain_policy: str,
            llm: str | None = None,
            llm_args: dict | None = None,
            **_: Any,
        ) -> _ClawLoopAgent:
            return _ClawLoopAgent(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm or "gpt-4o-mini",
                llm_args=llm_args or {},
            )

        _clawloop_agent_class = _ClawLoopAgent
        _clawloop_factory = _factory

    from tau2.registry import registry
    # Write directly to the private dict to support re-registration across
    # learning iterations. tau2's public register_agent_factory() raises
    # ValueError on duplicate names (tau2/registry.py ~L129) with no
    # overwrite=True option. Track https://github.com/sierra-research/tau2-bench
    # for a future public API that supports upsert.
    registry._agent_factories[_AGENT_KEY] = _clawloop_factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_step_boundaries(messages: list[Message]) -> list[int]:
    """Return indices of messages that start a new conversation step.

    A step starts at each user message that immediately follows a non-user
    message (or at the very first message). Consecutive user messages are
    treated as a single turn.
    """
    boundaries: list[int] = []
    for i, msg in enumerate(messages):
        if msg.role == "user" and (i == 0 or messages[i - 1].role != "user"):
            boundaries.append(i)
    if not boundaries and messages:
        boundaries = [0]
    return boundaries
