"""Task environment — the interface for plugging in benchmarks.

Users provide tasks (``Sample`` instances) and a scoring function;
ClawLoop handles everything else.  ``TaskEnvironment`` is the Protocol that
custom environments implement, and ``StaticTaskEnvironment`` is a
ready-made dataclass for the common case of a fixed task list with
an external evaluate function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

# ---------------------------------------------------------------------------
# Sample — a single evaluation task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A single task presented to the agent for evaluation.

    ``question`` is the prompt text.  ``context`` provides optional
    background, ``ground_truth`` is the expected answer (may be ``None``
    for open-ended tasks), and ``metadata`` carries arbitrary extra info.
    """

    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvalResult — the outcome of scoring one response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating an agent response against a ``Sample``.

    ``score`` is the primary metric (higher is better).  ``feedback``
    is a human-readable explanation, and ``metrics`` holds any
    additional numeric measurements.
    """

    score: float
    feedback: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TaskEnvironment — Protocol
# ---------------------------------------------------------------------------


class TaskEnvironment(Protocol):
    """Protocol that custom benchmark environments implement.

    ``get_tasks()`` returns the list of samples to evaluate, and
    ``evaluate(sample, response)`` scores a single agent response.
    """

    def get_tasks(self) -> list[Sample]:
        """Return all tasks in this environment."""
        ...

    def evaluate(self, sample: Sample, response: str) -> EvalResult:
        """Score *response* against *sample* and return an ``EvalResult``."""
        ...


# ---------------------------------------------------------------------------
# StaticTaskEnvironment — ready-made implementation
# ---------------------------------------------------------------------------


@dataclass
class StaticTaskEnvironment:
    """A ``TaskEnvironment`` backed by a fixed list of tasks and an
    external scoring callable.

    This covers the common case where the task set is known up front
    and scoring logic is a simple function.
    """

    tasks: list[Sample]
    evaluate_fn: Callable[[Sample, str], EvalResult]

    def get_tasks(self) -> list[Sample]:
        """Return the stored task list."""
        return self.tasks

    def evaluate(self, sample: Sample, response: str) -> EvalResult:
        """Delegate to ``evaluate_fn``."""
        return self.evaluate_fn(sample, response)
