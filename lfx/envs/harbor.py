"""Harbor environment adapter — runs Harbor trials, produces LfX Episodes."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.types import SampleContext
from lfx.utils.async_bridge import run_async

if TYPE_CHECKING:
    from lfx.core.loop import AgentState

log = logging.getLogger(__name__)


class HarborTaskEnvironment:
    """Runs Harbor trials and produces LfX Episodes. Harbor is optional."""

    def __init__(self, task_dir: Path, trial_config: dict,
                 reward_transform: Callable[[float], float] | None = None,
                 train_on_truncated: bool = True):
        try:
            from harbor.models.trial.config import TrialConfig
            from harbor.trial.trial import Trial
        except ImportError as exc:
            raise ImportError(
                "Harbor is required for HarborTaskEnvironment. "
                "Install it with: pip install lfx[harbor]"
            ) from exc
        self._Trial = Trial
        self._TrialConfig = TrialConfig
        self._task_dir = Path(task_dir)
        self._trial_config = trial_config
        self._reward_transform = reward_transform
        self._train_on_truncated = train_on_truncated
        if "agent" not in trial_config:
            raise ValueError("trial_config must contain 'agent' key")
        trial_config.setdefault("task", {})
        trial_config["agent"].setdefault("kwargs", {})

    @property
    def task_id(self) -> str:
        return self._task_dir.name

    async def run_episode(self, agent_state: AgentState) -> Episode:
        config = deepcopy(self._trial_config)
        config["task"]["path"] = str(self._task_dir)
        config["agent"]["kwargs"]["session_id"] = uuid4().hex

        if hasattr(agent_state, "inference_url") and agent_state.inference_url:
            config["agent"]["kwargs"]["api_base"] = agent_state.inference_url

        if hasattr(agent_state, "harness") and agent_state.harness:
            try:
                # Try task-specific bench first, fall back to "harbor" bench
                sample_result = agent_state.harness.sample(SampleContext(bench=self._task_dir.name))
                prompt = sample_result.result().output
                if not prompt:
                    sample_result = agent_state.harness.sample(SampleContext(bench="harbor"))
                    prompt = sample_result.result().output
                if prompt:  # Only override when harness returns a non-empty prompt
                    config["agent"]["kwargs"]["system_prompt_override"] = prompt
            except Exception:
                log.debug("Failed to sample system prompt from harness", exc_info=True)

        trial = self._Trial(self._TrialConfig(**config))
        try:
            results = await trial.run()
        except Exception as e:
            exc_name = type(e).__name__
            if exc_name == "ContextLengthExceededError":
                if self._train_on_truncated:
                    return self._build_episode(agent_state, reward=0.0, metadata={"truncated": True})
                else:
                    return self._build_episode(agent_state, filtered=True, metadata={"truncated": True})
            elif exc_name == "AgentTimeoutError":
                return self._build_episode(agent_state, filtered=True, metadata={"timeout": True})
            else:
                return self._build_episode(agent_state, filtered=True, metadata={"error": exc_name})

        if results.verifier_result is None or results.verifier_result.rewards is None:
            chat_history = []
            if results.agent_result and results.agent_result.metadata:
                chat_history = results.agent_result.metadata.get("all_messages", [])
            return self._build_episode(agent_state, chat_history=chat_history,
                                       reward=0.0, metadata={"verifier_none": True})

        raw_reward = results.verifier_result.rewards.get("reward", 0.0)
        metadata: dict[str, Any] = {"raw_reward": raw_reward}
        try:
            reward = self._reward_transform(raw_reward) if self._reward_transform else raw_reward
        except Exception:
            reward = raw_reward
            metadata["reward_transform_error"] = True
        metadata["transformed_reward"] = reward

        chat_history = []
        if results.agent_result and results.agent_result.metadata:
            chat_history = results.agent_result.metadata.get("all_messages", [])
        score_breakdown = results.verifier_result.rewards

        return self._build_episode(agent_state, chat_history=chat_history, reward=reward,
                                   score_breakdown=score_breakdown, metadata=metadata)

    def _build_episode(self, agent_state: AgentState, chat_history=None, reward=0.0,
                       filtered=False, score_breakdown=None, metadata=None) -> Episode:
        from lfx.core.reward import RewardSignal

        messages = [Message(role=m.get("role", "user"), content=m.get("content", ""))
                    for m in (chat_history or []) if isinstance(m, dict)]
        step_boundaries = _compute_step_boundaries(messages)
        steps = _build_steps(step_boundaries, reward)
        summary = EpisodeSummary(filtered=filtered, score_breakdown=score_breakdown)
        if not filtered:
            if self._reward_transform is not None:
                # Transformed reward is already in the caller's target range.
                # Set signal directly — RewardSignal clamps to [-1, 1].
                summary.signals["outcome"] = RewardSignal(
                    name="outcome", value=float(reward), confidence=1.0,
                )
            else:
                # Raw Harbor reward is [0, 1]. total_reward setter maps to [-1, 1].
                summary.total_reward = float(reward)
        state_id = ""
        if hasattr(agent_state, "state_id") and callable(agent_state.state_id):
            try:
                state_id = agent_state.state_id().combined_hash
            except Exception:
                log.debug("Failed to compute state_id for episode", exc_info=True)
        return Episode(
            id=uuid4().hex, state_id=state_id or "", task_id=self.task_id,
            bench="harbor", messages=messages, step_boundaries=step_boundaries,
            steps=steps, summary=summary, metadata=metadata or {},
        )


def _compute_step_boundaries(messages: list[Message]) -> list[int]:
    boundaries = []
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


class HarborAdapter:
    """Sync wrapper around HarborTaskEnvironment list. Implements AdapterLike."""

    def __init__(self, envs: list[HarborTaskEnvironment]):
        self._envs: dict[str, HarborTaskEnvironment] = {}
        for env in envs:
            if env.task_id in self._envs:
                raise ValueError(
                    f"Duplicate task_id {env.task_id!r} — "
                    f"each HarborTaskEnvironment must have a unique task directory name"
                )
            self._envs[env.task_id] = env

    def run_episode(self, task: str, agent_state: AgentState) -> Episode:
        return run_async(self._envs[task].run_episode(agent_state))

    def run_batch(self, agent_state: AgentState, tasks: list[str], n_per_task: int = 1) -> list[Episode]:
        async def _gather():
            coros = [self._envs[t].run_episode(agent_state) for t in tasks for _ in range(n_per_task)]
            return await asyncio.gather(*coros)
        return run_async(_gather())
