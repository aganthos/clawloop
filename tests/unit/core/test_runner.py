"""Unit tests for ``clawloop.core.runner.EpisodeCollectorRunner``."""

from __future__ import annotations

import random
from typing import Any

import pytest

from clawloop.core.runner import EpisodeCollectorRunner


def _mk_episode(task_id: str) -> Any:
    """Sentinel stand-in for Episode. Runner never inspects episodes — it
    only passes them through from adapter to caller — so a plain object
    tagged with ``task_id`` is sufficient and keeps the test decoupled
    from Episode's evolving required fields.
    """
    return {"task_id": task_id}


class _PerTaskAdapter:
    """Adapter exposing only ``run_episode`` (fallback path)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def run_episode(self, task: Any, agent_state: Any) -> Any:
        self.calls.append((task, agent_state))
        return _mk_episode(str(task))


class _BatchAdapter:
    """Adapter exposing ``run_batch`` (fast path)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, list[Any]]] = []

    def run_batch(self, agent_state: Any, tasks: list[Any]) -> list[Any]:
        self.calls.append((agent_state, list(tasks)))
        return [_mk_episode(str(t)) for t in tasks]


def test_empty_tasks_returns_empty_without_touching_adapter() -> None:
    adapter = _PerTaskAdapter()
    runner = EpisodeCollectorRunner(adapter)
    assert runner.collect(agent_state=object(), tasks=[], n_episodes=3) == []
    assert adapter.calls == []


def test_zero_n_episodes_returns_empty_without_touching_adapter() -> None:
    adapter = _PerTaskAdapter()
    runner = EpisodeCollectorRunner(adapter)
    assert runner.collect(agent_state=object(), tasks=["a", "b"], n_episodes=0) == []
    assert adapter.calls == []


def test_negative_n_episodes_returns_empty() -> None:
    adapter = _PerTaskAdapter()
    runner = EpisodeCollectorRunner(adapter)
    assert runner.collect(agent_state=object(), tasks=["a"], n_episodes=-1) == []
    assert adapter.calls == []


def test_sample_branch_uses_sample_without_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_sample(pool: list[Any], k: int) -> list[Any]:
        calls["sample"] = (list(pool), k)
        return pool[:k]

    def fake_choices(pool: list[Any], k: int) -> list[Any]:  # pragma: no cover — must not run
        raise AssertionError("random.choices should not be called when n <= len(tasks)")

    monkeypatch.setattr(random, "sample", fake_sample)
    monkeypatch.setattr(random, "choices", fake_choices)

    adapter = _PerTaskAdapter()
    runner = EpisodeCollectorRunner(adapter)
    episodes = runner.collect(agent_state="S", tasks=["a", "b", "c"], n_episodes=2)

    assert [ep["task_id"] for ep in episodes] == ["a", "b"]
    assert calls["sample"] == (["a", "b", "c"], 2)


def test_choices_branch_used_when_n_exceeds_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_sample(pool: list[Any], k: int) -> list[Any]:  # pragma: no cover
        raise AssertionError("random.sample should not be called when n > len(tasks)")

    def fake_choices(pool: list[Any], *, k: int) -> list[Any]:
        calls["choices"] = (list(pool), k)
        return [pool[0]] * k

    monkeypatch.setattr(random, "sample", fake_sample)
    monkeypatch.setattr(random, "choices", fake_choices)

    adapter = _PerTaskAdapter()
    runner = EpisodeCollectorRunner(adapter)
    episodes = runner.collect(agent_state="S", tasks=["x"], n_episodes=3)

    assert len(episodes) == 3
    assert calls["choices"] == (["x"], 3)


def test_prefers_run_batch_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "sample", lambda pool, k: list(pool[:k]))

    adapter = _BatchAdapter()
    runner = EpisodeCollectorRunner(adapter)
    agent = object()
    episodes = runner.collect(agent_state=agent, tasks=["a", "b", "c"], n_episodes=2)

    assert len(episodes) == 2
    assert len(adapter.calls) == 1
    called_agent, called_tasks = adapter.calls[0]
    assert called_agent is agent
    assert called_tasks == ["a", "b"]


def test_per_task_fallback_when_no_run_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "sample", lambda pool, k: list(pool[:k]))

    adapter = _PerTaskAdapter()
    runner = EpisodeCollectorRunner(adapter)
    agent = object()
    episodes = runner.collect(agent_state=agent, tasks=["a", "b"], n_episodes=2)

    assert [ep["task_id"] for ep in episodes] == ["a", "b"]
    assert [agent_state for _, agent_state in adapter.calls] == [agent, agent]


def test_non_callable_run_batch_attribute_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-callable ``run_batch`` attribute must not short-circuit the fallback."""
    monkeypatch.setattr(random, "sample", lambda pool, k: list(pool[:k]))

    class _WeirdAdapter:
        run_batch = "not-a-callable"

        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []

        def run_episode(self, task: Any, agent_state: Any) -> Any:
            self.calls.append((task, agent_state))
            return _mk_episode(str(task))

    adapter = _WeirdAdapter()
    runner = EpisodeCollectorRunner(adapter)
    episodes = runner.collect(agent_state="S", tasks=["a"], n_episodes=1)

    assert [ep["task_id"] for ep in episodes] == ["a"]
    assert adapter.calls == [("a", "S")]
