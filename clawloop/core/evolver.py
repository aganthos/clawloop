"""Internal Evolver interface — pluggable harness optimization backends.

NOT a public protocol. The external API is the Layer Protocol
(forward_backward/optim_step). This is the internal contract that
different optimization strategies implement within the Harness.

Layer Protocol = transport boundary (lifecycle).
Evolver = implementation boundary (algorithm).
Management methods on Harness = introspection boundary (richness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from clawloop.core.episode import Episode
from clawloop.learning_layers.harness import Insight, PromptCandidate


# ---------------------------------------------------------------------------
# Harness state snapshot (serializable for cloud evolvers)
# ---------------------------------------------------------------------------

@dataclass
class HarnessSnapshot:
    """Complete harness state for an Evolver to analyze."""

    system_prompts: dict[str, str]
    playbook_entries: list[dict[str, Any]]
    pareto_fronts: dict[str, list[dict[str, Any]]]
    playbook_generation: int
    playbook_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompts": self.system_prompts,
            "playbook_entries": self.playbook_entries,
            "pareto_fronts": self.pareto_fronts,
            "playbook_generation": self.playbook_generation,
            "playbook_version": self.playbook_version,
        }


# ---------------------------------------------------------------------------
# Evolver context and result
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """Metadata about who produced this result and at what cost."""

    backend: str = ""
    version: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    seed: int | None = None


@dataclass
class EvolverContext:
    """Context beyond the current episode batch."""

    reward_history: list[float] = field(default_factory=list)
    is_stagnating: bool = False
    iteration: int = 0
    tried_paradigms: list[str] = field(default_factory=list)
    max_tokens: int | None = None
    max_candidates: int | None = None


@dataclass
class EvolverResult:
    """What an Evolver returns — can touch all three harness mechanisms.

    For synchronous (local) evolvers: fully populated, run_id empty.
    For async (cloud) evolvers: may be partial, run_id set for polling.
    """

    insights: list[Insight] = field(default_factory=list)
    candidates: dict[str, list[PromptCandidate]] = field(default_factory=dict)
    paradigm_shift: bool = False
    deprecation_targets: list[str] = field(default_factory=list)
    run_id: str = ""
    provenance: Provenance = field(default_factory=Provenance)


# ---------------------------------------------------------------------------
# Evolver interface (internal, not exported as public API)
# ---------------------------------------------------------------------------

class Evolver(Protocol):
    """Internal interface for harness optimization backends.

    Receives episode traces + full harness state, returns holistic
    improvements across playbook, prompts, and paradigm.

    Implementations: LocalEvolver and other optimization backends.
    """

    def evolve(
        self,
        episodes: list[Episode],
        harness_state: HarnessSnapshot,
        context: EvolverContext,
    ) -> EvolverResult: ...

    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Standardized FBResult.info schema
# ---------------------------------------------------------------------------

_INFO_VERSION = 1

# Valid lifecycle statuses for FBResult.info["status"]:
#   ok        — evolution complete, results in pending state
#   running   — long-running evolution in progress (cloud backends)
#   paused    — waiting for user input (interactive candidate selection)
#   failed    — evolution failed, see info["error"]
#   cancelled — evolution was cancelled via harness.cancel()
VALID_STATUSES = ("ok", "running", "paused", "failed", "cancelled")


def make_fb_info(
    *,
    status: str = "ok",
    run_id: str = "",
    summary: str = "",
    candidates_tested: int = 0,
    best_score: float | None = None,
    archive_size: int = 0,
    paradigm_shifted: bool = False,
    backend: str = "",
    tokens_used: int = 0,
    progress: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Build a standardized FBResult.info dict.

    Schema is versioned via info_version so clients can evolve.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}, must be one of {VALID_STATUSES}")
    info: dict[str, Any] = {
        "info_version": _INFO_VERSION,
        "status": status,
        "run_id": run_id,
    }
    if summary:
        info["summary"] = summary
    if candidates_tested:
        info["candidates_tested"] = candidates_tested
    if best_score is not None:
        info["best_score"] = best_score
    if archive_size:
        info["archive_size"] = archive_size
    if paradigm_shifted:
        info["paradigm_shifted"] = True
    if backend:
        info["backend"] = backend
    if tokens_used:
        info["tokens_used"] = tokens_used
    if progress is not None:
        info["progress"] = progress
    if error:
        info["error"] = error
    return info
