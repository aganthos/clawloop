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

import copy
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from clawloop.core.types import (
    Datum, FBResult, Future, LoadResult, OptimResult,
    SampleContext, SampleResult, SaveResult,
)

import logging
import re

log = logging.getLogger(__name__)

# Max content length for insights (character count).
_MAX_INSIGHT_CONTENT_LENGTH = 2000

# Patterns that may indicate prompt injection attempts in insight content.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
]

# Whitelist for insight tag characters (alphanumeric, hyphens, underscores).
_TAG_RE = re.compile(r"^[a-zA-Z0-9\-_]+$")


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
    source_episode_ids: list[str] = field(default_factory=list)
    # --- Structured skill fields ---
    name: str = ""
    description: str = ""
    anti_patterns: str = ""
    category: str = "general"
    # --- Temporal / decay ---
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    generation: int = 0
    decay_rate: float = 0.01
    # --- Embedding cache ---
    embedding: list[float] | None = None
    embedding_model_id: str | None = None
    embedding_updated_at: float | None = None
    # --- Supersession ---
    superseded_by: str | None = None

    @staticmethod
    def new_id(prefix: str = "str") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def score(self) -> float:
        """Net usefulness signal (helpful - harmful)."""
        return float(self.helpful - self.harmful)

    def effective_score(self) -> float:
        """Net score with temporal decay applied."""
        anchor = self.last_activated if self.last_activated != self.created_at else self.created_at
        age_days = (time.time() - anchor) / 86400
        raw = float(self.helpful - self.harmful)
        return raw * math.exp(-self.decay_rate * age_days)

    def needs_reembed(self, current_model_id: str) -> bool:
        """True if embedding is missing, stale, or from a different model."""
        if self.embedding is None:
            return True
        if self.embedding_model_id != current_model_id:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "tags": self.tags,
            "source_episode_ids": self.source_episode_ids,
            "name": self.name,
            "description": self.description,
            "anti_patterns": self.anti_patterns,
            "category": self.category,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "generation": self.generation,
            "decay_rate": self.decay_rate,
            "superseded_by": self.superseded_by,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
            d["embedding_model_id"] = self.embedding_model_id
            d["embedding_updated_at"] = self.embedding_updated_at
        return d


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

    def prune_by_effective_score(self, min_score: float = 0.0) -> int:
        """Remove entries whose effective_score (with decay) is below threshold."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.effective_score() >= min_score]
        return before - len(self.entries)

    def active_entries(self) -> list[PlaybookEntry]:
        """Return entries that are not superseded."""
        return [e for e in self.entries if not e.superseded_by]

    def render(self, tags: set[str] | None = None) -> str:
        """Render the playbook as a text block for inclusion in system prompts.

        When *tags* is provided, only entries whose tags overlap with the
        given set are included (DC-RS / ACE-style selective retrieval).
        Falls back to all active entries if no tags match.
        """
        entries = self.active_entries()
        if tags:
            matched = [e for e in entries if e.tags and set(e.tags) & tags]
            if matched:
                entries = matched
        if not entries:
            return ""
        lines = ["## PLAYBOOK"]
        for e in entries:
            if e.name and e.description:
                lines.append(f"### {e.name}")
                lines.append(f"**When**: {e.description}")
                lines.append(e.content)
                if e.anti_patterns:
                    lines.append(f"**Anti-pattern**: {e.anti_patterns}")
            else:
                tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
                lines.append(f"[{e.id}]{tag_str} :: {e.content}")
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


# -- Pending accumulator --


@dataclass
class _HarnessPending:
    """Accumulator for forward_backward signals. Drained by optim_step."""
    playbook_signals: dict[str, tuple[int, int]] = field(default_factory=dict)
    insights: list[Insight] = field(default_factory=list)
    candidates: dict[str, list[PromptCandidate]] = field(default_factory=dict)
    activated_entries: set[str] = field(default_factory=set)  # entry IDs to update last_activated
    deprecation_targets: list[str] = field(default_factory=list)  # entry IDs for paradigm decay


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

    # Internal: pending signals accumulated by forward_backward, drained by optim_step
    _pending: _HarnessPending = field(default_factory=_HarnessPending)

    # Incremented on each successful optim_step that applies at least one update
    playbook_version: int = 0

    # Incremented on structural playbook changes (insights applied, not just score updates)
    playbook_generation: int = 0

    # Optional Evolver for unified optimization during forward_backward
    # (replaces the old reflector= slot — wraps Reflector + GEPA + Paradigm)
    evolver: Any | None = field(default=None, repr=False)

    # Optional PlaybookCurator for retrieve-classify-revise pipeline
    _curator: Any | None = field(default=None, repr=False)

    # Evolver context set by the loop before forward_backward
    _evolver_context: Any | None = field(default=None, repr=False)

    # Optional embedding provider for semantic playbook retrieval
    _embeddings: Any | None = field(default=None, repr=False)

    # Cosine similarity threshold for embedding retrieval
    _retrieval_threshold: float = 0.4

    # Max entries returned in full-playbook fallback
    _max_retrieval_entries: int = 50

    def system_prompt(
        self,
        bench: str,
        task_tags: set[str] | None = None,
        query_text: str | None = None,
    ) -> str:
        """Resolve the system prompt for a bench, including playbook.

        Retrieval priority (when *query_text* is provided):
          1. Embedding similarity (primary — semantic match)
          2. Tag filter (fallback — category match)
          3. Full playbook (final fallback — capped by effective score)

        Without *query_text*, uses the existing tag-based path unchanged.
        """
        base = self.system_prompts.get(bench, "")

        if query_text and query_text.strip() and self.embeddings is not None:
            entries, reason = self._retrieve_entries(task_tags, query_text.strip())
            pb = self._render_entries(entries, reason)
        else:
            pb = self.playbook.render(tags=task_tags)

        if pb:
            return f"{base}\n\n{pb}" if base else pb
        return base

    @property
    def embeddings(self) -> Any | None:
        """Return the embedding provider (explicit or from curator)."""
        if self._embeddings is not None:
            return self._embeddings
        if self._curator is not None and hasattr(self._curator, "_embeddings"):
            return self._curator._embeddings
        return None

    def _retrieve_entries(
        self,
        task_tags: set[str] | None,
        query_text: str,
    ) -> tuple[list[PlaybookEntry], str]:
        """Embedding-first retrieval with tag and full-playbook fallbacks."""
        active = self.playbook.active_entries()
        if not active:
            return [], "empty"

        # 1. Embedding retrieval (primary)
        hits = self._embed_and_find(query_text, active)
        if hits:
            # Cap results to prevent prompt bloat
            return hits[: self._max_retrieval_entries], "embedding"

        # 2. Tag filter (secondary fallback)
        if task_tags:
            tag_hits = [e for e in active if e.tags and set(e.tags) & task_tags]
            if tag_hits:
                return tag_hits, "tags"

        # 3. Full playbook (final fallback, capped by effective score)
        if len(active) > self._max_retrieval_entries:
            active = sorted(active, key=lambda e: e.effective_score(), reverse=True)
            active = active[: self._max_retrieval_entries]
        return active, "full"

    def _embed_and_find(
        self, query_text: str, entries: list[PlaybookEntry],
    ) -> list[PlaybookEntry]:
        """Embed query and find similar entries. Returns [] on any failure."""
        provider = self.embeddings
        if provider is None:
            return []

        from clawloop.core.embeddings import find_similar

        # Lazy embed: entries missing or stale embeddings get refreshed
        model_id = getattr(provider, "model", None) or "unknown"
        needs = [e for e in entries if e.needs_reembed(model_id)]
        if needs:
            try:
                texts = [e.content for e in needs]
                vectors = provider.embed(texts)
                for entry, vec in zip(needs, vectors):
                    entry.embedding = vec
                    entry.embedding_model_id = model_id
                    entry.embedding_updated_at = time.time()
            except Exception:
                log.warning("Failed to embed playbook entries", exc_info=True)

        # Embed query
        query_text = query_text[:500]
        try:
            query_vec = provider.embed([query_text])[0]
        except Exception:
            log.warning("Failed to embed query text", exc_info=True)
            return []

        try:
            results = find_similar(query_vec, entries, threshold=self._retrieval_threshold)
            return [entry for entry, _score in results]
        except Exception:
            log.warning("find_similar failed", exc_info=True)
            return []

    @staticmethod
    def _render_entries(entries: list[PlaybookEntry], reason: str) -> str:
        """Render retrieved entries with a header indicating retrieval method."""
        if not entries:
            return ""
        header = "## PLAYBOOK (semantic match)" if reason == "embedding" else "## PLAYBOOK"
        lines = [header]
        for e in entries:
            if e.name and e.description:
                lines.append(f"### {e.name}")
                lines.append(f"**When**: {e.description}")
                lines.append(e.content)
                if e.anti_patterns:
                    lines.append(f"**Anti-pattern**: {e.anti_patterns}")
            else:
                tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
                lines.append(f"[{e.id}]{tag_str} :: {e.content}")
        return "\n".join(lines)

    def apply_insights(self, insights: list[Insight]) -> int:
        """Curator step: apply reflector insights as playbook deltas.

        Returns the number of deltas applied.  All insights are validated
        through :meth:`_validate_insights` before application — this gates
        ALL insight sources including :class:`ParadigmBreakthrough`.

        When a curator is configured, "add" insights are routed through the
        retrieve-classify-revise pipeline instead of direct append.

        .. warning:: Security — indirect prompt injection

            Insights originate from LLM-based Reflectors that analyse episode
            traces.  A malicious or adversarial episode could cause the
            Reflector to emit a poisoned insight that persists in the playbook
            and therefore in all future system prompts.  Before using this in
            production, callers MUST gate insights through a validation layer
            (schema check, content policy filter, or human-in-the-loop
            approval).  See PR #1 review for context.
        """
        insights = self._validate_insights(insights)
        applied = 0
        structural_change = False
        for insight in insights:
            if insight.action == "add":
                if self._curator is not None:
                    result = self._curator.curate_insight(insight, self.playbook)
                    log.debug(
                        "Curator: %s (affected=%s)",
                        result.action, result.entries_affected,
                    )
                    if result.action != "skip_redundant":
                        structural_change = True
                        applied += 1
                else:
                    entry = PlaybookEntry(
                        id=PlaybookEntry.new_id(),
                        content=insight.content,
                        tags=insight.tags,
                        source_episode_ids=list(insight.source_episode_ids),
                    )
                    self.playbook.add(entry)
                    structural_change = True
                    applied += 1
            elif insight.action == "update" and insight.target_entry_id:
                existing = self.playbook.lookup(insight.target_entry_id)
                if existing:
                    existing.content = insight.content
                    existing.helpful += 1
                    # Invalidate cached embedding on content change
                    existing.embedding = None
                    existing.embedding_model_id = None
                    existing.embedding_updated_at = None
                    applied += 1
                    structural_change = True
            elif insight.action == "remove" and insight.target_entry_id:
                if self.playbook.remove(insight.target_entry_id):
                    applied += 1
                    structural_change = True

        # Increment playbook_generation on structural changes
        if structural_change:
            self.playbook_generation += 1

        return applied

    @staticmethod
    def _validate_insights(insights: list[Insight]) -> list[Insight]:
        """Filter insights that fail basic safety checks.

        Guards against indirect prompt injection by rejecting insights whose
        content is suspiciously long, matches known injection patterns, or
        has invalid structure.
        """
        safe: list[Insight] = []
        for insight in insights:
            # Action validity (already enforced by Insight.__post_init__, but
            # defend against manually-constructed dicts)
            if insight.action not in Insight.VALID_ACTIONS:
                log.warning("Dropping insight — invalid action %r", insight.action)
                continue

            # update/remove require target_entry_id
            if insight.action in ("update", "remove") and not insight.target_entry_id:
                log.warning(
                    "Dropping insight — %s requires target_entry_id", insight.action,
                )
                continue

            # Type checks
            if not isinstance(insight.content, str):
                log.warning("Dropping insight — content must be str")
                continue
            if not isinstance(insight.tags, list) or not all(
                isinstance(t, str) for t in insight.tags
            ):
                log.warning("Dropping insight — tags must be list[str]")
                continue
            if hasattr(insight, "source_episode_ids") and insight.source_episode_ids:
                if not isinstance(insight.source_episode_ids, list) or not all(
                    isinstance(s, str) for s in insight.source_episode_ids
                ):
                    log.warning("Dropping insight — source_episode_ids must be list[str]")
                    continue

            # Tag character whitelist
            if any(not _TAG_RE.match(t) for t in insight.tags if t):
                log.warning("Dropping insight — tags contain invalid characters")
                continue

            # Content length
            if len(insight.content) > _MAX_INSIGHT_CONTENT_LENGTH:
                log.warning(
                    "Dropping insight (length %d exceeds %d)",
                    len(insight.content),
                    _MAX_INSIGHT_CONTENT_LENGTH,
                )
                continue

            # Injection patterns (content + tags)
            if any(p.search(insight.content) for p in _INJECTION_PATTERNS) or any(
                p.search(tag) for tag in insight.tags for p in _INJECTION_PATTERNS
            ):
                log.warning("Dropping insight — matches injection pattern")
                continue

            safe.append(insight)
        return safe

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
            # Check coherence with playbook
            if self._curator is not None and self.playbook.entries:
                try:
                    conflicts = self._curator.check_prompt_playbook_coherence(
                        best.text, self.playbook,
                    )
                    if conflicts:
                        log.warning(
                            "GEPA-Playbook conflicts detected: %s", conflicts,
                        )
                except Exception:
                    log.debug("Coherence check failed", exc_info=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompts": dict(self.system_prompts),
            "pareto_fronts": {
                k: v.to_dict() for k, v in self.pareto_fronts.items()
            },
            "playbook": self.playbook.to_dict(),
            "tool_configs": [tc.to_dict() for tc in self.tool_configs],
            "validators": {
                k: getattr(v, "name", v.__class__.__name__)
                for k, v in self.validators.items()
            },
            "playbook_version": self.playbook_version,
            "playbook_generation": self.playbook_generation,
        }

    # -- Layer protocol methods --

    def _attribute_entries(self, episode: Any) -> list[PlaybookEntry]:
        """Return playbook entries relevant to this episode.

        Attribution strategy (cheapest first):
        1. Tag match: if episode has tags/bench, match against entry tags.
        2. Embedding similarity: cosine sim between entry and episode content.

        Falls back to all entries if no attribution method works.
        """
        active = self.playbook.active_entries()
        if not active:
            return []

        # Collect episode tags for matching
        ep_tags: set[str] = set()
        if hasattr(episode, "bench") and episode.bench:
            ep_tags.add(episode.bench)
        if hasattr(episode, "metadata") and isinstance(episode.metadata, dict):
            for t in episode.metadata.get("tags", []):
                ep_tags.add(str(t))

        # Strategy 1: Tag match
        if ep_tags:
            tag_matched = [
                e for e in active
                if e.tags and ep_tags & set(e.tags)
            ]
            if tag_matched:
                return tag_matched

        # Strategy 2: Embedding similarity (if curator has embeddings)
        if self._curator is not None:
            try:
                from clawloop.core.embeddings import cosine_similarity
                # Build a simple text representation of the episode
                ep_text = " ".join(
                    m.content for m in episode.messages
                    if m.role in ("user", "assistant") and m.content
                )[:500]
                if ep_text:
                    ep_embedding = self._curator._embeddings.embed([ep_text])[0]
                    ep_dim = len(ep_embedding)
                    current_model = getattr(self._curator._embeddings, "model", None)
                    relevant = []
                    for entry in active:
                        if entry.embedding is None:
                            continue
                        # Skip entries with dimension mismatch (stale model)
                        if len(entry.embedding) != ep_dim:
                            continue
                        # Skip entries with stale embedding model
                        if current_model and entry.needs_reembed(current_model):
                            continue
                        sim = cosine_similarity(ep_embedding, entry.embedding)
                        if sim >= (self._curator._config.attribution_threshold):
                            relevant.append(entry)
                    if relevant:
                        return relevant
            except Exception:
                log.debug("Embedding attribution failed — falling back to all entries")

        # Fallback: all active entries
        return active

    # -- Evolver helpers --

    def set_evolver_context(self, ctx: Any) -> None:
        """Set the EvolverContext for the next forward_backward call."""
        self._evolver_context = ctx

    def _build_snapshot(self) -> Any:
        """Build a HarnessSnapshot from current state for the Evolver."""
        from clawloop.core.evolver import HarnessSnapshot

        return HarnessSnapshot(
            system_prompts=dict(self.system_prompts),
            playbook_entries=[e.to_dict() for e in self.playbook.entries],
            pareto_fronts={
                bench: [c.to_dict() for c in front.candidates]
                for bench, front in self.pareto_fronts.items()
            },
            playbook_generation=self.playbook_generation,
            playbook_version=self.playbook_version,
        )

    # -- Management methods (stubs for local, rich for cloud) --

    def evolution_summary(self, run_id: str = "") -> dict[str, Any]:
        """Return summary of last/current evolution."""
        return {"backend": self.evolver.name() if self.evolver else "none"}

    def get_candidates(self, bench: str = "") -> list[dict[str, Any]]:
        """Return current Pareto front candidates for inspection."""
        if bench and bench in self.pareto_fronts:
            return [
                {"text": c.text, "scores": c.per_task_scores}
                for c in self.pareto_fronts[bench].candidates
            ]
        return []

    def cancel(self, run_id: str = "") -> bool:
        """Cancel a running evolution. No-op for local evolvers."""
        return False

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Analyse episodes and accumulate signals without mutating observable state.

        For each episode, attributes rewards only to relevant playbook entries
        (tag match or embedding similarity). Skips stale episodes whose
        ``scored_at_generation`` is behind the current ``playbook_generation``.
        """
        stale_skipped = 0
        for episode in data.episodes:
            # Skip stale episodes (scored before current playbook generation)
            gen = getattr(episode.summary, "scored_at_generation", None)
            if gen is not None and gen < self.playbook_generation:
                stale_skipped += 1
                continue

            reward = episode.summary.effective_reward()  # [-1, 1]
            if reward == 0:
                continue

            # Only attribute to relevant entries
            relevant_entries = self._attribute_entries(episode)
            for entry in relevant_entries:
                prev_h, prev_harm = self._pending.playbook_signals.get(
                    entry.id, (0, 0)
                )
                if reward > 0:
                    self._pending.playbook_signals[entry.id] = (prev_h + 1, prev_harm)
                else:
                    self._pending.playbook_signals[entry.id] = (prev_h, prev_harm + 1)
                # Defer last_activated update to optim_step
                self._pending.activated_entries.add(entry.id)

        metrics: dict[str, Any] = {
            "episodes_processed": len(data.episodes),
            "entries_signaled": len(self._pending.playbook_signals),
            "stale_skipped": stale_skipped,
        }

        # Evolver: unified optimization (reflector + GEPA + paradigm).
        if self.evolver is not None:
            try:
                from clawloop.core.evolver import EvolverContext, make_fb_info

                snapshot = self._build_snapshot()
                ctx = self._evolver_context or EvolverContext()
                evolver_result = self.evolver.evolve(data.episodes, snapshot, ctx)

                # Merge evolver results into pending
                self._pending.insights.extend(evolver_result.insights)
                for bench, cands in evolver_result.candidates.items():
                    self._pending.candidates.setdefault(bench, []).extend(cands)
                if evolver_result.deprecation_targets:
                    self._pending.deprecation_targets.extend(evolver_result.deprecation_targets)

                metrics["insights_generated"] = len(evolver_result.insights)
                metrics["candidates_generated"] = sum(
                    len(c) for c in evolver_result.candidates.values()
                )
                metrics["paradigm_shifted"] = evolver_result.paradigm_shift

                fb_info = make_fb_info(
                    status="ok",
                    run_id=evolver_result.run_id,
                    candidates_tested=metrics["candidates_generated"],
                    paradigm_shifted=evolver_result.paradigm_shift,
                    backend=evolver_result.provenance.backend,
                )
                metrics.update(fb_info)
            except Exception:
                log.exception("Evolver failed during forward_backward")
            finally:
                # Clear context to prevent stale reuse on next call
                self._evolver_context = None

        return Future.immediate(FBResult(status="ok", metrics=metrics))

    def optim_step(self) -> Future[OptimResult]:
        """Apply accumulated signals with snapshot-rollback for atomicity.

        Uses deepcopy to snapshot playbook, system_prompts, and pareto_fronts
        before applying.  On failure, rolls back to the snapshot.
        """
        pending = self._pending
        has_signals = bool(pending.playbook_signals)
        has_insights = bool(pending.insights)
        has_candidates = bool(pending.candidates)
        has_deprecation = bool(pending.deprecation_targets)

        if not (has_signals or has_insights or has_candidates or has_deprecation):
            return Future.immediate(OptimResult(status="ok", updates_applied=0))

        # Snapshot for rollback
        snap_playbook = copy.deepcopy(self.playbook)
        snap_system_prompts = copy.deepcopy(self.system_prompts)
        snap_pareto_fronts = copy.deepcopy(self.pareto_fronts)

        try:
            updates = 0

            # Apply playbook signals
            now = time.time()
            for entry_id, (h_delta, harm_delta) in pending.playbook_signals.items():
                entry = self.playbook.lookup(entry_id)
                if entry is not None:
                    entry.helpful += h_delta
                    entry.harmful += harm_delta
                    if entry_id in pending.activated_entries:
                        entry.last_activated = now
                    updates += 1

            # Apply pending insights (apply_insights validates internally)
            if pending.insights:
                updates += self.apply_insights(pending.insights)

            # Apply pending Pareto candidates
            for bench, candidates in pending.candidates.items():
                for candidate in candidates:
                    self.update_pareto(bench, candidate)
                    updates += 1

            # Apply paradigm deprecation: increase decay on targeted entries
            if pending.deprecation_targets:
                deprecation_decay = 0.05
                target_set = set(pending.deprecation_targets)
                for entry in self.playbook.entries:
                    if entry.id in target_set:
                        entry.decay_rate = max(entry.decay_rate, deprecation_decay)
                        updates += 1

            # Auto-prune entries with sustained negative score
            # Only prune entries that have enough signal (helpful+harmful >= 3)
            before_prune = len(self.playbook.entries)
            self.playbook.entries = [
                e for e in self.playbook.entries
                if e.score() >= 0.0 or (e.helpful + e.harmful) < 3
            ]
            pruned = before_prune - len(self.playbook.entries)
            if pruned:
                log.info("Auto-pruned %d low-scoring playbook entries", pruned)
                updates += pruned
                self.playbook_generation += 1

            # Hard cap on active entries (exclude superseded from count)
            max_entries = 100
            if self._curator is not None:
                max_entries = self._curator.max_entries
            active = self.playbook.active_entries()
            if len(active) > max_entries:
                active.sort(key=lambda e: e.effective_score(), reverse=True)
                to_remove = active[max_entries:]
                for entry in to_remove:
                    self.playbook.remove(entry.id)
                overflow = len(to_remove)
                log.info(
                    "Capped playbook at %d active entries (removed %d)",
                    max_entries, overflow,
                )
                updates += overflow
                self.playbook_generation += 1

            # Drain pending
            self._pending = _HarnessPending()
            if updates > 0:
                self.playbook_version += 1
            return Future.immediate(OptimResult(status="ok", updates_applied=updates))

        except Exception:
            # Rollback
            self.playbook = snap_playbook
            self.system_prompts = snap_system_prompts
            self.pareto_fronts = snap_pareto_fronts
            self._pending = _HarnessPending()
            return Future.immediate(OptimResult(status="error", updates_applied=0))

    def clear_pending_state(self) -> None:
        """Reset the internal pending accumulator."""
        self._pending = _HarnessPending()

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Return the resolved system prompt for the given bench."""
        output = self.system_prompt(ctx.bench)
        return Future.immediate(SampleResult(output=output))

    def save_state(self, name: str) -> Future[SaveResult]:
        """Persist a named checkpoint (actual storage is handled externally)."""
        return Future.immediate(SaveResult(name=name, status="ok"))

    def load_state(self, state_dict: dict) -> Future[LoadResult]:
        """Restore harness state from a serialized dict.

        Rebuilds system_prompts, playbook (from entries), pareto_fronts
        (from candidate dicts), tool_configs, and validators.  Clears _pending.
        """
        self.system_prompts = state_dict.get("system_prompts", {})

        # Restore playbook
        pb_data = state_dict.get("playbook", {})
        entries = [
            PlaybookEntry(
                id=e["id"],
                content=e["content"],
                helpful=e.get("helpful", 0),
                harmful=e.get("harmful", 0),
                tags=e.get("tags", []),
                source_episode_ids=e.get("source_episode_ids", []),
                name=e.get("name", ""),
                description=e.get("description", ""),
                anti_patterns=e.get("anti_patterns", ""),
                category=e.get("category", "general"),
                created_at=e.get("created_at", time.time()),
                last_activated=e.get("last_activated", time.time()),
                generation=e.get("generation", 0),
                decay_rate=e.get("decay_rate", 0.01),
                embedding=e.get("embedding"),
                embedding_model_id=e.get("embedding_model_id"),
                embedding_updated_at=e.get("embedding_updated_at"),
                superseded_by=e.get("superseded_by"),
            )
            for e in pb_data.get("entries", [])
        ]
        self.playbook = Playbook(entries=entries)

        # Restore pareto fronts
        pf_data = state_dict.get("pareto_fronts", {})
        self.pareto_fronts = {}
        for bench, front_dict in pf_data.items():
            candidates = [
                PromptCandidate(
                    id=c["id"],
                    text=c["text"],
                    per_task_scores=c.get("per_task_scores", {}),
                    generation=c.get("generation", 0),
                    parent_id=c.get("parent_id"),
                )
                for c in front_dict.get("candidates", [])
            ]
            self.pareto_fronts[bench] = ParetoFront(candidates=candidates)

        # Restore tool configs
        tc_data = state_dict.get("tool_configs", [])
        self.tool_configs = [
            ToolConfig(
                name=tc["name"],
                schema=tc["schema"],
                owner=tc["owner"],
                mutable=tc["mutable"],
                sandbox_required=tc.get("sandbox_required", False),
            )
            for tc in tc_data
        ]

        # Restore validators
        self.validators = state_dict.get("validators", {})

        # Restore version counters
        self.playbook_version = state_dict.get("playbook_version", 0)
        self.playbook_generation = state_dict.get("playbook_generation", 0)

        # Clear pending
        self._pending = _HarnessPending()

        return Future.immediate(LoadResult(status="ok"))
