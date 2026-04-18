"""Dataclass schema for evolution archive records."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class RunRecord:
    """Top-level record for one evolution run."""

    run_id: str
    bench: str
    domain_tags: list[str]
    agent_config: dict[str, Any]
    config_hash: str
    n_iterations: int
    best_reward: float
    improvement_delta: float
    total_cost_tokens: int
    parent_run_id: str | None
    created_at: float
    completed_at: float | None

    @staticmethod
    def new_id() -> str:
        """Generate a fresh run ID."""
        return uuid.uuid4().hex

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "bench": self.bench,
            "domain_tags": self.domain_tags,
            "agent_config": self.agent_config,
            "config_hash": self.config_hash,
            "n_iterations": self.n_iterations,
            "best_reward": self.best_reward,
            "improvement_delta": self.improvement_delta,
            "total_cost_tokens": self.total_cost_tokens,
            "parent_run_id": self.parent_run_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        return cls(
            run_id=d["run_id"],
            bench=d["bench"],
            domain_tags=d["domain_tags"],
            agent_config=d["agent_config"],
            config_hash=d["config_hash"],
            n_iterations=d["n_iterations"],
            best_reward=d["best_reward"],
            improvement_delta=d["improvement_delta"],
            total_cost_tokens=d["total_cost_tokens"],
            parent_run_id=d.get("parent_run_id"),
            created_at=d["created_at"],
            completed_at=d.get("completed_at"),
        )


@dataclass
class IterationRecord:
    """One iteration within an evolution run."""

    run_id: str
    iteration_num: int
    harness_snapshot_hash: str
    mean_reward: float
    reward_trajectory: list[float]
    evolver_action: dict[str, Any]
    cost_tokens: int
    parent_variant_hash: str | None
    child_variant_hash: str | None
    reward_delta: float
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "iteration_num": self.iteration_num,
            "harness_snapshot_hash": self.harness_snapshot_hash,
            "mean_reward": self.mean_reward,
            "reward_trajectory": self.reward_trajectory,
            "evolver_action": self.evolver_action,
            "cost_tokens": self.cost_tokens,
            "parent_variant_hash": self.parent_variant_hash,
            "child_variant_hash": self.child_variant_hash,
            "reward_delta": self.reward_delta,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IterationRecord:
        return cls(
            run_id=d["run_id"],
            iteration_num=d["iteration_num"],
            harness_snapshot_hash=d["harness_snapshot_hash"],
            mean_reward=d["mean_reward"],
            reward_trajectory=d["reward_trajectory"],
            evolver_action=d["evolver_action"],
            cost_tokens=d["cost_tokens"],
            parent_variant_hash=d.get("parent_variant_hash"),
            child_variant_hash=d.get("child_variant_hash"),
            reward_delta=d["reward_delta"],
            created_at=d["created_at"],
        )


@dataclass
class EpisodeRecord:
    """One episode (single task execution) within an iteration."""

    run_id: str
    iteration_num: int
    episode_id: str
    task_id: str
    bench: str
    model: str
    reward: float
    signals: dict[str, Any]
    n_steps: int
    n_tool_calls: int
    token_usage: dict[str, Any]
    latency_ms: int
    messages_ref: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "iteration_num": self.iteration_num,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "bench": self.bench,
            "model": self.model,
            "reward": self.reward,
            "signals": self.signals,
            "n_steps": self.n_steps,
            "n_tool_calls": self.n_tool_calls,
            "token_usage": self.token_usage,
            "latency_ms": self.latency_ms,
            "messages_ref": self.messages_ref,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeRecord:
        return cls(
            run_id=d["run_id"],
            iteration_num=d["iteration_num"],
            episode_id=d["episode_id"],
            task_id=d["task_id"],
            bench=d["bench"],
            model=d["model"],
            reward=d["reward"],
            signals=d["signals"],
            n_steps=d["n_steps"],
            n_tool_calls=d["n_tool_calls"],
            token_usage=d["token_usage"],
            latency_ms=d["latency_ms"],
            messages_ref=d["messages_ref"],
            created_at=d["created_at"],
        )


@dataclass
class AgentVariant:
    """Content-addressed snapshot of a specific agent configuration."""

    variant_hash: str
    system_prompt: str
    playbook_snapshot: dict[str, Any]
    model: str
    tools: list[str]
    first_seen_run_id: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_hash": self.variant_hash,
            "system_prompt": self.system_prompt,
            "playbook_snapshot": self.playbook_snapshot,
            "model": self.model,
            "tools": self.tools,
            "first_seen_run_id": self.first_seen_run_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentVariant:
        return cls(
            variant_hash=d["variant_hash"],
            system_prompt=d["system_prompt"],
            playbook_snapshot=d["playbook_snapshot"],
            model=d["model"],
            tools=d["tools"],
            first_seen_run_id=d["first_seen_run_id"],
            created_at=d["created_at"],
        )
