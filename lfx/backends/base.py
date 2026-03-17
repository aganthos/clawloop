"""LfXBackend protocol, BackendError, and SkyRLBackendInitError.

BackendError is a frozen dataclass (not an Exception) that carries a
structured error code, human-readable message, and a recoverability flag.

SkyRLBackendInitError is a real Exception that wraps a BackendError and
is raised when a SkyRL backend cannot be initialised.

LfXBackend is a runtime-checkable Protocol identical to the existing Layer
protocol in lfx/core/layer.py — backends ARE layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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


# ---------------------------------------------------------------------------
# BackendError
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackendError:
    """Structured, immutable error descriptor for backend failures.

    ``code`` is one of the known error codes:
    - ``gpu_oom`` — GPU ran out of memory (recoverable)
    - ``tokenizer_mismatch`` — tokenizer schema mismatch
    - ``backend_unreachable`` — network/RPC failure (recoverable)
    - ``invalid_config`` — bad configuration value
    - ``training_diverged`` — loss NaN / divergence detected
    - ``schema_incompatible`` — type or attribute mismatch
    - ``import_error`` — missing Python package
    - ``unknown`` — unmapped exception
    """

    code: str
    message: str
    recoverable: bool

    @classmethod
    def from_exception(cls, e: Exception) -> BackendError:
        """Map a Python exception to a BackendError.

        Mapping priority:
        1. Exact exception type checks
        2. String content of the error message
        3. Default ``unknown`` fallback
        """
        msg = str(e)

        # -- Type-based mappings --
        if isinstance(e, MemoryError):
            return cls(code="gpu_oom", message=msg, recoverable=True)
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            return cls(code="import_error", message=msg, recoverable=False)
        if isinstance(e, (ConnectionError, TimeoutError)):
            return cls(code="backend_unreachable", message=msg, recoverable=True)
        if isinstance(e, (TypeError, AttributeError)):
            return cls(code="schema_incompatible", message=msg, recoverable=False)

        # -- String-content checks --
        lower = msg.lower()
        if "nan" in lower or "diverge" in lower:
            return cls(code="training_diverged", message=msg, recoverable=False)
        if "config" in lower or "invalid" in lower:
            return cls(code="invalid_config", message=msg, recoverable=False)

        return cls(code="unknown", message=msg, recoverable=False)


# ---------------------------------------------------------------------------
# SkyRLBackendInitError
# ---------------------------------------------------------------------------

class SkyRLBackendInitError(Exception):
    """Raised when a SkyRL backend cannot be initialised.

    Wraps a :class:`BackendError` for structured introspection.
    """

    def __init__(self, error: BackendError) -> None:
        self.error = error
        super().__init__(f"[{error.code}] {error.message}")


# ---------------------------------------------------------------------------
# LfXBackend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LfXBackend(Protocol):
    """Unified protocol for every LfX backend.

    Intentionally identical to the existing ``Layer`` protocol in
    ``lfx.core.layer`` — backends ARE layers and must satisfy the same
    two-phase forward_backward / optim_step contract.
    """

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Accumulate gradients / diffs from *data* without mutating state."""
        ...

    def optim_step(self) -> Future[OptimResult]:
        """Apply the accumulated updates atomically."""
        ...

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Produce backend-specific output for the given context."""
        ...

    def save_state(self, name: str) -> Future[SaveResult]:
        """Persist the current backend state under *name*."""
        ...

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        """Restore backend state from *state*."""
        ...

    def clear_pending_state(self) -> None:
        """Reset the internal pending accumulator."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise the backend configuration for hashing and logging."""
        ...
