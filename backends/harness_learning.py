"""HarnessLearningBackend — thin ClawLoopBackend wrapper around the Harness layer.

Pure delegation: every method forwards to the underlying Harness instance.
HarnessLearningConfig is a placeholder for future unified-mode parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clawloop.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)
from clawloop.layers.harness import Harness


@dataclass
class HarnessLearningConfig:
    """Placeholder config for unified mode. Currently unused by the backend."""

    reflector_enabled: bool = True
    intensity_config: dict[str, Any] = field(default_factory=dict)
    paradigm_enabled: bool = True


class HarnessLearningBackend:
    """Wraps the Harness layer as an ClawLoopBackend. Pure delegation."""

    def __init__(
        self,
        harness: Harness,
        config: HarnessLearningConfig | None = None,
    ) -> None:
        self._harness = harness
        self._config = config or HarnessLearningConfig()

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        return self._harness.forward_backward(data)

    def optim_step(self) -> Future[OptimResult]:
        return self._harness.optim_step()

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        return self._harness.sample(ctx)

    def save_state(self, name: str) -> Future[SaveResult]:
        return self._harness.save_state(name)

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        return self._harness.load_state(state)

    def clear_pending_state(self) -> None:
        self._harness.clear_pending_state()

    def to_dict(self) -> dict[str, Any]:
        return self._harness.to_dict()
