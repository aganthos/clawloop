"""Layer protocol — the unified interface for every learning layer.

Every layer in LfX implements this two-phase contract:

1. **Accumulate phase** — ``forward_backward(data)`` computes gradients /
   diffs / proposals from a batch of episodes and stores them internally.
   No parameters are mutated yet.

2. **Apply phase** — ``optim_step()`` atomically applies the accumulated
   updates (e.g. weight update, prompt rewrite, routing-table revision).

Additional verbs:
- ``sample(ctx)`` — produce layer-specific output for a given context
  (e.g. system prompt, routing decision, LoRA adapter).
- ``save_state(name)`` / ``load_state(state_dict)`` — checkpoint management.
- ``to_dict()`` — serialise the layer configuration for hashing / logging.

All mutating verbs return ``Future[ResultT]`` so callers can choose between
blocking and async-style usage.
"""

from __future__ import annotations

from typing import Any, Protocol

from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)


class Layer(Protocol):
    """Unified protocol for every learning layer in LfX.

    Implementers must provide the six methods below.  The two-phase
    contract (``forward_backward`` then ``optim_step``) ensures that
    gradient computation and parameter mutation are cleanly separated,
    enabling safe rollback and multi-layer coordination.
    """

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Accumulate gradients / diffs from *data* without mutating state."""
        ...

    def optim_step(self) -> Future[OptimResult]:
        """Apply the accumulated updates atomically."""
        ...

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Produce layer-specific output for the given context."""
        ...

    def save_state(self, name: str) -> Future[SaveResult]:
        """Persist the current layer state under *name*."""
        ...

    def load_state(self, state_dict: dict[str, Any]) -> Future[LoadResult]:
        """Restore layer state from *state_dict*."""
        ...

    def clear_pending_state(self) -> None:
        """Reset the internal pending accumulator.

        Called by the learning loop on the ``forward_backward`` error path
        to prevent partially-accumulated deltas from leaking into future
        iterations.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise the layer configuration for hashing and logging."""
        ...
