"""Core types for the ClawLoop Layer protocol.

Provides ``Future[T]`` for async results, ``Datum`` as the standard input
bundle, and lightweight result dataclasses returned by each Layer verb.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from clawloop.core.episode import Episode

T = TypeVar("T")

_UNSET = object()


class Future(Generic[T]):
    """A simple, thread-safe, single-assignment future.

    Used as the return type of every ``Layer`` verb so that callers can
    choose between blocking (``.result()``) and polling (``.done``).
    """

    def __init__(self) -> None:
        self._value: T = _UNSET  # type: ignore[assignment]
        self._event = threading.Event()
        self._lock = threading.Lock()

    # -- public API --

    def result(self, timeout: float | None = None) -> T:
        """Block until the value is available, then return it.

        Raises ``TimeoutError`` if *timeout* seconds elapse first.
        """
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("Future was not resolved within the timeout")
        return self._value

    def set_result(self, value: T) -> None:
        """Resolve the future with *value*.

        Raises ``RuntimeError`` if the future has already been resolved.
        """
        with self._lock:
            if self._event.is_set():
                raise RuntimeError("Future already resolved")
            self._value = value
            self._event.set()

    @property
    def done(self) -> bool:
        """Return ``True`` if the future has been resolved."""
        return self._event.is_set()

    # -- convenience --

    @classmethod
    def immediate(cls, value: T) -> Future[T]:
        """Create a ``Future`` that is already resolved with *value*."""
        f: Future[T] = cls()
        f.set_result(value)
        return f


# ---------------------------------------------------------------------------
# Datum — standard input bundle for forward_backward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Datum:
    """A batch of episodes plus loss-function configuration.

    ``episodes`` is the list of trajectories to learn from.
    ``loss_fn`` selects the loss function (``"auto"`` picks a sensible
    default for the layer).
    """

    episodes: list[Episode]
    loss_fn: str = "auto"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result dataclasses — one per Layer verb
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FBResult:
    """Result of ``Layer.forward_backward``."""

    status: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimResult:
    """Result of ``Layer.optim_step``."""

    status: str
    updates_applied: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleContext:
    """Input context for ``Layer.sample``."""

    bench: str = ""
    query_features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleResult:
    """Result of ``Layer.sample``."""

    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SaveResult:
    """Result of ``Layer.save_state``."""

    name: str = ""
    status: str = "ok"


@dataclass(frozen=True)
class LoadResult:
    """Result of ``Layer.load_state``."""

    status: str = "ok"
