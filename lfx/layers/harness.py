"""Harness layer — prompt optimization (GEPA), memory (ACE playbook), and tool configs.

This layer controls everything *except* which model is called (Router) and the
model weights themselves (Weights).

Learning strategy combines three mechanisms:
  1. **GEPA-style prompt evolution**: Reflective mutation with Pareto-front
     selection.  An LLM reads full execution traces as *Actionable Side
     Information* (ASI), diagnoses failures, and proposes targeted prompt fixes.
     A population of non-dominated candidates is maintained to prevent premature
     convergence.
  2. **ACE-style playbook memory**: A structured, itemized playbook where each
     entry carries helpful/harmful counters.  Updates are incremental deltas
     (never full rewrites) via a Generator->Reflector->Curator pipeline,
     preventing context collapse and brevity bias.
  3. **Tool config tuning**: Tool schemas and ownership can be evolved using
     the same GEPA reflective loop — optimizing descriptions so the agent
     selects the right tools more reliably.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


# -- Tool configuration --


@dataclass
class ToolConfig:
    """Configuration for a single tool available to the agent."""

    name: str
    schema: dict[str, Any]
    owner: str  # "env" | "harness"
    mutable: bool
    sandbox_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "owner": self.owner,
            "mutable": self.mutable,
            "sandbox_required": self.sandbox_required,
        }


# -- ACE playbook --


@dataclass
class PlaybookEntry:
    """One item in the ACE-style playbook.

    Each entry is a reusable strategy, domain concept, or failure-mode
    description.  Helpful/harmful counters track empirical value across
    episodes so the Curator can prune or reinforce entries.
    """

    id: str
    content: str
    helpful: int = 0
    harmful: int = 0
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def new_id(prefix: str = "str") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def score(self) -> float:
        """Net usefulness signal (helpful - harmful)."""
        return float(self.helpful - self.harmful)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "tags": self.tags,
        }


@dataclass
class Playbook:
    """ACE-style structured memory — a living document of agent strategies.

    Entries are organized by category (strategies, formulas, failure modes).
    Updates are always incremental deltas — never full rewrites — to prevent
    context collapse and brevity bias.
    """

    entries: list[PlaybookEntry] = field(default_factory=list)

    def lookup(self, entry_id: str) -> PlaybookEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def add(self, entry: PlaybookEntry) -> None:
        """Grow phase: append a new insight."""
        self.entries.append(entry)

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found."""
        for i, e in enumerate(self.entries):
            if e.id == entry_id:
                self.entries.pop(i)
                return True
        return False

    def prune(self, min_score: float = 0.0) -> int:
        """Remove entries whose net score (helpful - harmful) is below threshold."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.score() >= min_score]
        return before - len(self.entries)

    def render(self) -> str:
        """Render the playbook as a text block for inclusion in system prompts."""
        if not self.entries:
            return ""
        lines = ["## PLAYBOOK"]
        for e in self.entries:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(
                f"[{e.id}] helpful={e.helpful} harmful={e.harmful}{tag_str} :: {e.content}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}


# -- GEPA prompt candidate --


@dataclass
class PromptCandidate:
    """One candidate in the GEPA Pareto front.

    GEPA maintains a population of non-dominated prompt variants.  Each
    candidate tracks per-task scores so that Pareto selection can preserve
    candidates that excel on *some* task instances even if their average is
    suboptimal.
    """

    id: str
    text: str
    per_task_scores: dict[str, float] = field(default_factory=dict)
    generation: int = 0  # evolutionary generation
    parent_id: str | None = None

    @staticmethod
    def new_id() -> str:
        return f"pc-{uuid.uuid4().hex[:8]}"

    def mean_score(self) -> float:
        if not self.per_task_scores:
            return 0.0
        return sum(self.per_task_scores.values()) / len(self.per_task_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "per_task_scores": self.per_task_scores,
            "generation": self.generation,
            "parent_id": self.parent_id,
        }


@dataclass
class ParetoFront:
    """GEPA-style Pareto front of prompt candidates.

    A candidate survives if it is non-dominated: there is no other candidate
    that beats it on *every* task instance.  This diversification prevents
    premature convergence to local optima.
    """

    candidates: list[PromptCandidate] = field(default_factory=list)

    def add(self, candidate: PromptCandidate) -> None:
        self.candidates.append(candidate)
        self._prune_dominated()

    def best(self) -> PromptCandidate | None:
        """Return the candidate with the highest mean score."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.mean_score())

    def _prune_dominated(self) -> None:
        """Remove dominated candidates from the front."""
        if len(self.candidates) <= 1:
            return
        survivors: list[PromptCandidate] = []
        for c in self.candidates:
            dominated = False
            for other in self.candidates:
                if other is c:
                    continue
                if _dominates(other, c):
                    dominated = True
                    break
            if not dominated:
                survivors.append(c)
        self.candidates = survivors if survivors else self.candidates[:1]

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [c.to_dict() for c in self.candidates]}


def _dominates(a: PromptCandidate, b: PromptCandidate) -> bool:
    """True if *a* is strictly better than *b* on every shared task."""
    shared = set(a.per_task_scores) & set(b.per_task_scores)
    if not shared:
        return False
    return all(
        a.per_task_scores[t] >= b.per_task_scores[t] for t in shared
    ) and any(a.per_task_scores[t] > b.per_task_scores[t] for t in shared)


# -- Insight (Reflector output) --


@dataclass
class Insight:
    """Actionable Side Information (ASI) produced by the ACE Reflector.

    Each insight captures one lesson from episode analysis: what worked,
    what failed, and why.  The Curator converts these into PlaybookEntry
    deltas.
    """

    VALID_ACTIONS = frozenset({"add", "update", "remove"})

    content: str
    source_episode_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    action: str = "add"  # "add" | "update" | "remove"
    target_entry_id: str | None = None  # for update/remove

    def __post_init__(self) -> None:
        if self.action not in self.VALID_ACTIONS:
            raise ValueError(
                f"Invalid insight action {self.action!r}, "
                f"must be one of {sorted(self.VALID_ACTIONS)}"
            )


# -- Harness layer --


@dataclass
class Harness:
    """Prompt optimization, memory, tool configs, and validators.

    Learning mechanisms:
      - ``pareto_fronts``: per-bench GEPA prompt evolution
      - ``playbook``: ACE-style structured memory with delta updates
      - ``tool_configs``: tool schemas (mutable ones evolve via GEPA)
    """

    # Active system prompts (best from Pareto front, or manual)
    system_prompts: dict[str, str] = field(default_factory=dict)

    # GEPA Pareto fronts — per bench
    pareto_fronts: dict[str, ParetoFront] = field(default_factory=dict)

    # ACE playbook
    playbook: Playbook = field(default_factory=Playbook)

    # Tool definitions with ownership
    tool_configs: list[ToolConfig] = field(default_factory=list)

    # Output validators
    validators: dict[str, Any] = field(default_factory=dict)

    def system_prompt(self, bench: str) -> str:
        """Resolve the system prompt for a bench, including playbook."""
        base = self.system_prompts.get(bench, "")
        pb = self.playbook.render()
        if pb:
            return f"{base}\n\n{pb}" if base else pb
        return base

    def apply_insights(self, insights: list[Insight]) -> int:
        """Curator step: apply reflector insights as playbook deltas.

        Returns the number of deltas applied.

        .. warning:: Security — indirect prompt injection

            Insights originate from LLM-based Reflectors that analyse episode
            traces.  A malicious or adversarial episode could cause the
            Reflector to emit a poisoned insight that persists in the playbook
            and therefore in all future system prompts.  Before using this in
            production, callers MUST gate insights through a validation layer
            (schema check, content policy filter, or human-in-the-loop
            approval).  See PR #1 review for context.
        """
        applied = 0
        for insight in insights:
            if insight.action == "add":
                entry = PlaybookEntry(
                    id=PlaybookEntry.new_id(),
                    content=insight.content,
                    tags=insight.tags,
                )
                self.playbook.add(entry)
                applied += 1
            elif insight.action == "update" and insight.target_entry_id:
                existing = self.playbook.lookup(insight.target_entry_id)
                if existing:
                    existing.content = insight.content
                    existing.helpful += 1
                    applied += 1
            elif insight.action == "remove" and insight.target_entry_id:
                if self.playbook.remove(insight.target_entry_id):
                    applied += 1
        return applied

    def update_pareto(
        self, bench: str, candidate: PromptCandidate
    ) -> None:
        """Add a candidate to the bench's Pareto front."""
        if bench not in self.pareto_fronts:
            self.pareto_fronts[bench] = ParetoFront()
        self.pareto_fronts[bench].add(candidate)
        # Promote best candidate to active system prompt
        best = self.pareto_fronts[bench].best()
        if best:
            self.system_prompts[bench] = best.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompts": self.system_prompts,
            "pareto_fronts": {
                k: v.to_dict() for k, v in self.pareto_fronts.items()
            },
            "playbook": self.playbook.to_dict(),
            "tool_configs": [tc.to_dict() for tc in self.tool_configs],
            "validators": self.validators,
        }
