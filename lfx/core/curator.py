"""PlaybookCurator — retrieve-classify-revise pipeline for playbook curation.

Before blindly adding every insight to the playbook, the Curator runs a
three-phase pipeline:

1. **RETRIEVE** — embed the insight, find similar existing entries.
2. **CLASSIFY** — heuristics first (cheap), then LLM if ambiguous.
3. **REVISE** — add / merge / resolve conflict / skip redundant.

A periodic ``consolidate()`` pass ("dreaming") clusters similar entries,
merges clusters, prunes negative-score entries, and enforces a hard cap.

Tasks 1.3 + 1.6 + 1.11 from the playbook curator spec.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lfx.core.embeddings import EmbeddingProvider, cosine_similarity, find_similar
from lfx.layers.harness import Insight, Playbook, PlaybookEntry

if TYPE_CHECKING:
    from lfx.llm import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contradiction keywords for heuristic conflict detection
# ---------------------------------------------------------------------------

CONTRADICTION_KEYWORDS: list[str] = [
    "not",
    "never",
    "avoid",
    "don't",
    "instead",
    "rather than",
    "opposite",
    "contrary",
    "however",
    "but",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CuratorConfig:
    """Tunable thresholds and caps for the curation pipeline."""

    identical_threshold: float = 0.95  # cosine sim for auto-classify as identical
    conflict_threshold: float = 0.8  # cosine sim for conflict check
    similar_threshold: float = 0.6  # min cosine sim for LLM classification
    attribution_threshold: float = 0.4  # min relevance for attribution
    max_playbook_entries: int = 100  # hard cap
    consolidation_interval: int = 10  # every N optim_steps
    cluster_threshold: float = 0.7  # for agglomerative clustering


@dataclass
class CurationResult:
    """Outcome of curating a single insight."""

    action: str  # "add" | "merge" | "conflict_resolved" | "skip_redundant"
    entries_affected: list[str]  # IDs
    new_entry: PlaybookEntry | None


@dataclass
class ConsolidationReport:
    """Summary of a consolidation ("dreaming") pass."""

    before: int
    after: int
    merged: int
    pruned: int
    conflicts_resolved: int


@dataclass
class CuratorMetrics:
    """Cumulative counters for curator operations."""

    insights_processed: int = 0
    added: int = 0
    skipped_redundant: int = 0
    merged: int = 0
    conflicts_resolved: int = 0
    fallback_direct_adds: int = 0
    consolidation_runs: int = 0
    entries_pruned: int = 0
    stale_episodes_skipped: int = 0


# ---------------------------------------------------------------------------
# PlaybookCurator
# ---------------------------------------------------------------------------


class PlaybookCurator:
    """Retrieve-classify-revise pipeline for principled playbook curation.

    Instead of naively appending every insight, the curator checks semantic
    similarity against existing entries to decide: add, merge, resolve a
    conflict, or skip a redundant insight.

    Periodic ``consolidate()`` calls cluster and compress the playbook.
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider | None = None,
        llm: LLMClient | None = None,
        config: CuratorConfig | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._llm = llm
        self._config = config or CuratorConfig()
        self._metrics = CuratorMetrics()

    @classmethod
    def lightweight(cls, max_entries: int = 50) -> PlaybookCurator:
        """Create a curator without embeddings or LLM.

        Suitable for narrow agents (e.g. n8n workflows) where playbooks
        are small enough that all entries fit in context. Provides pruning
        and capping but no similarity-based dedup or merging.
        """
        return cls(config=CuratorConfig(max_playbook_entries=max_entries))

    @property
    def metrics(self) -> CuratorMetrics:
        """Read-only access to cumulative metrics."""
        return self._metrics

    @property
    def max_entries(self) -> int:
        """Maximum number of active playbook entries."""
        return self._config.max_playbook_entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curate_insight(
        self, insight: Insight, playbook: Playbook
    ) -> CurationResult:
        """Run the retrieve-classify-revise pipeline for a single insight.

        Falls back to a direct add if embedding or LLM calls fail.
        """
        self._metrics.insights_processed += 1

        try:
            return self._curate_insight_inner(insight, playbook)
        except Exception:
            log.warning(
                "Curator pipeline failed — falling back to direct add",
                exc_info=True,
            )
            return self._fallback_add(insight, playbook)

    def consolidate(self, playbook: Playbook) -> ConsolidationReport:
        """Periodic 'dreaming' pass — cluster, merge, prune.

        1. Cluster active entries by embedding similarity (agglomerative).
        2. Within each cluster of >1 entry: LLM merges into one.
        3. Prune entries with effective_score < 0.
        4. Cap at max_entries by effective_score.

        Without embeddings/LLM, only prune and cap steps run.
        """
        self._metrics.consolidation_runs += 1
        before = len(playbook.entries)
        merged_count = 0
        conflicts_resolved = 0

        # --- Clustering + merging requires embeddings + LLM ---
        if self._embeddings is not None and self._llm is not None:
            self._ensure_embeddings(playbook)

            # --- 1. Cluster ---
            active = playbook.active_entries()
            clusters = self._cluster_entries(active)
        else:
            clusters = []

        # --- 2. Merge multi-entry clusters ---
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            try:
                merged_entry = self._merge_cluster(cluster)
                # Remove originals from playbook, add merged
                for entry in cluster:
                    entry.superseded_by = merged_entry.id
                playbook.add(merged_entry)
                merged_count += len(cluster)
            except Exception:
                log.warning(
                    "Failed to merge cluster of %d entries — skipping",
                    len(cluster),
                    exc_info=True,
                )

        # --- 3. Prune negative effective_score entries ---
        pruned = playbook.prune_by_effective_score(min_score=0.0)
        self._metrics.entries_pruned += pruned

        # --- 4. Cap at max entries ---
        cap_pruned = self._cap_entries(playbook)
        pruned += cap_pruned
        self._metrics.entries_pruned += cap_pruned

        after = len(playbook.entries)

        return ConsolidationReport(
            before=before,
            after=after,
            merged=merged_count,
            pruned=pruned,
            conflicts_resolved=conflicts_resolved,
        )

    def check_prompt_playbook_coherence(
        self, prompt_text: str, playbook: Playbook
    ) -> list[str]:
        """Check for conflicts between a GEPA prompt and playbook entries.

        Returns a list of human-readable conflict descriptions (empty if
        no conflicts detected).
        """
        if not playbook.entries:
            return []

        active = playbook.active_entries()
        if not active:
            return []

        if self._llm is None:
            return []

        # Build a prompt for the LLM to find conflicts
        entries_text = "\n".join(
            f"- [{e.id}] {e.content}" for e in active
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a consistency checker. Identify conflicts between "
                    "the system prompt and playbook entries. A conflict means "
                    "the prompt instructs one behaviour while a playbook entry "
                    "instructs the opposite. Return a JSON array of strings, "
                    "each describing one conflict. Return [] if no conflicts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"SYSTEM PROMPT:\n{prompt_text}\n\n"
                    f"PLAYBOOK ENTRIES:\n{entries_text}\n\n"
                    "Conflicts (JSON array):"
                ),
            },
        ]

        try:
            result = self._llm.complete(messages)
            text = str(result).strip()
            # Parse JSON array from response
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return []
        except Exception:
            log.warning(
                "Coherence check failed — returning empty",
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _curate_insight_inner(
        self, insight: Insight, playbook: Playbook
    ) -> CurationResult:
        """Core retrieve-classify-revise logic (may raise)."""
        insight_text = insight.content

        # --- No embeddings: lightweight mode (tag-only, direct add) ---
        if self._embeddings is None:
            return self._add_new(insight, playbook, embedding=None)

        # --- RETRIEVE ---
        self._ensure_embeddings(playbook)
        embedding = self._embeddings.embed([insight_text])[0]

        active = playbook.active_entries()
        similar = find_similar(
            embedding,
            active,
            threshold=self._config.similar_threshold,
        )

        # No similar entries — just add
        if not similar:
            return self._add_new(insight, playbook, embedding)

        # --- CLASSIFY ---
        top_entry, top_sim = similar[0]

        # Try heuristic classification first
        classification = self._classify_heuristic(
            insight_text, top_entry, top_sim
        )

        # If heuristic is ambiguous, use LLM (if available)
        if classification is None:
            if self._llm is not None:
                classification = self._classify_llm(insight_text, similar)
            else:
                classification = "unrelated"

        # --- REVISE ---
        if classification == "identical":
            # Skip redundant — just bump helpful count on the existing entry
            top_entry.helpful += 1
            top_entry.last_activated = time.time()
            self._metrics.skipped_redundant += 1
            return CurationResult(
                action="skip_redundant",
                entries_affected=[top_entry.id],
                new_entry=None,
            )

        if classification == "conflicting":
            result = self._resolve_conflict(insight, similar, playbook)
            self._metrics.conflicts_resolved += 1
            return result

        if classification == "complementary":
            result = self._merge(insight, similar, playbook)
            self._metrics.merged += 1
            return result

        # "unrelated" or anything else — just add
        return self._add_new(insight, playbook, embedding)

    def _classify_heuristic(
        self,
        insight_text: str,
        entry: PlaybookEntry,
        similarity: float,
    ) -> str | None:
        """Heuristic classification. Returns None if ambiguous (needs LLM).

        Returns:
            "identical" — cosine sim > identical_threshold
            "conflicting" — cosine sim > conflict_threshold AND contradiction keywords
            None — ambiguous, needs LLM classification
        """
        if similarity >= self._config.identical_threshold:
            return "identical"

        if similarity >= self._config.conflict_threshold:
            # Check for contradiction keywords in either text
            insight_lower = insight_text.lower()
            entry_lower = entry.content.lower()

            # Count contradiction keywords present in the insight but not
            # the entry (or vice versa) — asymmetric presence suggests
            # the insight is negating/contradicting the existing entry.
            insight_has = sum(
                1 for kw in CONTRADICTION_KEYWORDS if kw in insight_lower
            )
            entry_has = sum(
                1 for kw in CONTRADICTION_KEYWORDS if kw in entry_lower
            )
            # If one side has notably more contradiction markers, it likely
            # contradicts the other.  Also flag if both have high counts
            # (both are negative instructions that may conflict).
            if abs(insight_has - entry_has) >= 2 or (
                insight_has >= 2 and entry_has >= 2
            ):
                return "conflicting"

        # Ambiguous — needs LLM
        if similarity >= self._config.similar_threshold:
            return None

        return "unrelated"

    def _classify_llm(
        self,
        insight_text: str,
        similar: list[tuple[PlaybookEntry, float]],
    ) -> str:
        """LLM classification for ambiguous cases.

        Returns one of: "identical", "complementary", "conflicting", "unrelated".
        """
        entries_text = "\n".join(
            f"- [{entry.id}] (sim={sim:.2f}) {entry.content}"
            for entry, sim in similar[:5]  # cap context to top 5
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge-base curator. Classify the relationship "
                    "between a new insight and existing playbook entries.\n\n"
                    "Respond with EXACTLY one word:\n"
                    '- "identical" — the insight says the same thing\n'
                    '- "complementary" — the insight adds to or refines existing entries\n'
                    '- "conflicting" — the insight contradicts existing entries\n'
                    '- "unrelated" — the insight covers a different topic\n\n'
                    "Respond with just the single classification word."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"NEW INSIGHT:\n{insight_text}\n\n"
                    f"EXISTING ENTRIES:\n{entries_text}\n\n"
                    "Classification:"
                ),
            },
        ]

        result = self._llm.complete(messages)
        classification = str(result).strip().lower().strip('"').strip("'")

        valid = {"identical", "complementary", "conflicting", "unrelated"}
        if classification not in valid:
            log.warning(
                "LLM returned unexpected classification %r — defaulting to 'unrelated'",
                classification,
            )
            return "unrelated"

        return classification

    def _create_merged_entry(
        self,
        content: str,
        source_entries: list[PlaybookEntry],
        extra_source_ids: list[str] | None = None,
        extra_tags: list[str] | None = None,
        helpful: int | None = None,
        harmful: int = 0,
        prefix: str = "cur",
    ) -> PlaybookEntry:
        """Create a new PlaybookEntry by aggregating metadata from source entries."""
        new_id = PlaybookEntry.new_id(prefix=prefix)

        source_ids: list[str] = list(extra_source_ids or [])
        for entry in source_entries:
            source_ids.extend(entry.source_episode_ids)

        all_tags: list[str] = list(extra_tags or [])
        for entry in source_entries:
            all_tags.extend(entry.tags)
        unique_tags = list(dict.fromkeys(all_tags))

        new_embedding = None
        if self._embeddings is not None:
            try:
                new_embedding = self._embeddings.embed([content])[0]
            except Exception:
                pass

        now = time.time()
        model_id = getattr(self._embeddings, "model", None) if self._embeddings else None
        if helpful is None:
            helpful = sum(e.helpful for e in source_entries) + 1

        return PlaybookEntry(
            id=new_id,
            content=content,
            helpful=helpful,
            harmful=harmful,
            tags=unique_tags,
            source_episode_ids=source_ids,
            created_at=now,
            last_activated=now,
            generation=max((e.generation for e in source_entries), default=0) + 1,
            embedding=new_embedding,
            embedding_model_id=model_id if new_embedding else None,
            embedding_updated_at=now if new_embedding else None,
        )

    def _supersede_and_add(
        self,
        new_entry: PlaybookEntry,
        originals: list[PlaybookEntry],
        playbook: Playbook,
        action: str,
    ) -> CurationResult:
        """Mark originals as superseded, add new entry, return result."""
        affected_ids: list[str] = []
        for entry in originals:
            entry.superseded_by = new_entry.id
            affected_ids.append(entry.id)
        playbook.add(new_entry)
        return CurationResult(
            action=action,
            entries_affected=affected_ids,
            new_entry=new_entry,
        )

    def _resolve_conflict(
        self,
        insight: Insight,
        similar: list[tuple[PlaybookEntry, float]],
        playbook: Playbook,
    ) -> CurationResult:
        """Create new entry that resolves the conflict, mark originals as superseded."""
        conflicting_entries = [entry for entry, _sim in similar[:3]]
        entries_text = "\n".join(f"- [{e.id}] {e.content}" for e in conflicting_entries)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge-base curator resolving a conflict. "
                    "A new insight contradicts existing playbook entries. "
                    "Write a single, clear, authoritative entry that supersedes "
                    "all conflicting entries. Incorporate the best information "
                    "from the new insight and existing entries. Keep the new "
                    "insight's perspective as it reflects more recent learning. "
                    "Respond with ONLY the text of the new entry — no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"NEW INSIGHT:\n{insight.content}\n\n"
                    f"CONFLICTING ENTRIES:\n{entries_text}\n\n"
                    "Resolved entry:"
                ),
            },
        ]

        resolved_text = str(self._llm.complete(messages)).strip()
        new_entry = self._create_merged_entry(
            resolved_text, conflicting_entries,
            extra_source_ids=list(insight.source_episode_ids),
            extra_tags=list(insight.tags),
            harmful=0,
        )
        return self._supersede_and_add(new_entry, conflicting_entries, playbook, "conflict_resolved")

    def _merge(
        self,
        insight: Insight,
        similar: list[tuple[PlaybookEntry, float]],
        playbook: Playbook,
    ) -> CurationResult:
        """Merge insight with similar entries into one stronger entry."""
        merge_candidates = [entry for entry, _sim in similar[:3]]
        entries_text = "\n".join(f"- [{e.id}] {e.content}" for e in merge_candidates)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge-base curator. Merge the new insight "
                    "with the existing complementary entries into a single, "
                    "comprehensive entry. Combine all useful information "
                    "without redundancy. Respond with ONLY the text of the "
                    "merged entry — no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"NEW INSIGHT:\n{insight.content}\n\n"
                    f"EXISTING ENTRIES:\n{entries_text}\n\n"
                    "Merged entry:"
                ),
            },
        ]

        merged_text = str(self._llm.complete(messages)).strip()
        new_entry = self._create_merged_entry(
            merged_text, merge_candidates,
            extra_source_ids=list(insight.source_episode_ids),
            extra_tags=list(insight.tags),
            harmful=sum(e.harmful for e in merge_candidates),
        )
        return self._supersede_and_add(new_entry, merge_candidates, playbook, "merge")

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    def _cluster_entries(
        self, entries: list[PlaybookEntry]
    ) -> list[list[PlaybookEntry]]:
        """Simple agglomerative clustering by embedding similarity.

        Uses single-linkage: two clusters merge if ANY pair of entries
        across them exceeds the cluster_threshold.

        Entries without embeddings are placed in singleton clusters.
        """
        threshold = self._config.cluster_threshold

        # Start: each entry in its own cluster
        clusters: list[list[PlaybookEntry]] = [[e] for e in entries]

        # Greedily merge the closest pair until no pair exceeds threshold
        changed = True
        while changed:
            changed = False
            best_sim = -1.0
            best_i = -1
            best_j = -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._max_cluster_similarity(clusters[i], clusters[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_i = i
                        best_j = j

            if best_sim >= threshold and best_i >= 0:
                # Merge cluster j into cluster i
                clusters[best_i].extend(clusters[best_j])
                clusters.pop(best_j)
                changed = True

        return clusters

    def _max_cluster_similarity(
        self,
        cluster_a: list[PlaybookEntry],
        cluster_b: list[PlaybookEntry],
    ) -> float:
        """Maximum cosine similarity between any pair across two clusters."""
        max_sim = -1.0
        for a in cluster_a:
            if a.embedding is None:
                continue
            for b in cluster_b:
                if b.embedding is None:
                    continue
                sim = cosine_similarity(a.embedding, b.embedding)
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    def _merge_cluster(self, cluster: list[PlaybookEntry]) -> PlaybookEntry:
        """LLM merges a cluster of similar entries into one.

        Raises on LLM failure — caller handles the exception.
        """
        entries_text = "\n".join(
            f"- [{e.id}] (score={e.effective_score():.1f}) {e.content}"
            for e in cluster
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge-base curator consolidating similar entries. "
                    "Merge the following entries into a single, clear, comprehensive "
                    "entry. Preserve all unique information. Remove redundancy. "
                    "Respond with ONLY the merged text — no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ENTRIES TO MERGE:\n{entries_text}\n\n"
                    "Merged entry:"
                ),
            },
        ]

        merged_text = str(self._llm.complete(messages)).strip()
        return self._create_merged_entry(
            merged_text, cluster,
            helpful=sum(e.helpful for e in cluster),
            harmful=sum(e.harmful for e in cluster),
            prefix="con",
        )

    def _cap_entries(self, playbook: Playbook) -> int:
        """Enforce max_playbook_entries by dropping lowest-scoring entries.

        Returns the number of entries pruned.
        """
        max_entries = self._config.max_playbook_entries
        active = playbook.active_entries()

        if len(active) <= max_entries:
            return 0

        # Sort by effective_score ascending — prune from the bottom
        active.sort(key=lambda e: e.effective_score())
        to_prune = len(active) - max_entries
        pruned = 0

        for entry in active[:to_prune]:
            if playbook.remove(entry.id):
                pruned += 1

        return pruned

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _ensure_embeddings(self, playbook: Playbook) -> None:
        """Embed any entries missing or stale embedding vectors.

        Batches all entries needing (re-)embedding into a single embed() call.
        Checks both missing embeddings and model ID mismatches.
        """
        current_model_id = getattr(self._embeddings, "model", None)
        needs_embed: list[PlaybookEntry] = [
            e for e in playbook.entries
            if e.embedding is None
            or (current_model_id is not None and e.needs_reembed(current_model_id))
        ]
        if not needs_embed:
            return

        texts = [e.content for e in needs_embed]
        try:
            embeddings = self._embeddings.embed(texts)
        except Exception:
            log.warning(
                "Batch embedding failed for %d entries", len(needs_embed),
                exc_info=True,
            )
            return

        now = time.time()
        model_id = getattr(self._embeddings, "model", None)
        for entry, emb in zip(needs_embed, embeddings):
            entry.embedding = emb
            entry.embedding_model_id = model_id
            entry.embedding_updated_at = now

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _add_new(
        self,
        insight: Insight,
        playbook: Playbook,
        embedding: list[float] | None = None,
    ) -> CurationResult:
        """Create and add a fresh PlaybookEntry from an insight."""
        new_id = PlaybookEntry.new_id(prefix="cur")
        now = time.time()
        model_id = getattr(self._embeddings, "model", None) if self._embeddings else None

        entry = PlaybookEntry(
            id=new_id,
            content=insight.content,
            helpful=1,
            tags=list(insight.tags),
            source_episode_ids=list(insight.source_episode_ids),
            created_at=now,
            last_activated=now,
            embedding=embedding,
            embedding_model_id=model_id if embedding else None,
            embedding_updated_at=now if embedding else None,
        )

        playbook.add(entry)
        self._metrics.added += 1

        return CurationResult(
            action="add",
            entries_affected=[],
            new_entry=entry,
        )

    def _fallback_add(
        self,
        insight: Insight,
        playbook: Playbook,
    ) -> CurationResult:
        """Emergency fallback: add directly, no classification."""
        self._metrics.fallback_direct_adds += 1
        return self._add_new(insight, playbook, embedding=None)
