"""LocalEvolver — community default wrapping Reflector + GEPA + Paradigm.

Synchronous evolver that delegates to the three existing harness learning
mechanisms. Always returns status="ok" with empty run_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from clawloop.core.episode import Episode
from clawloop.core.evolver import (
    EvolverContext,
    EvolverResult,
    HarnessSnapshot,
    Provenance,
)
from clawloop.learning_layers.harness import Insight, Playbook, PromptCandidate

log = logging.getLogger(__name__)


@dataclass
class LocalEvolver:
    """Wraps Reflector + PromptEvolver + ParadigmBreakthrough as one Evolver.

    Each sub-component is optional. If absent, that mechanism is skipped.
    """

    reflector: Any | None = None  # Reflector
    prompt_evolver: Any | None = None  # PromptEvolver (GEPA)
    paradigm: Any | None = None  # ParadigmBreakthrough

    def name(self) -> str:
        return "local"

    def evolve(
        self,
        episodes: list[Episode],
        harness_state: HarnessSnapshot,
        context: EvolverContext,
    ) -> EvolverResult:
        """Run all local mechanisms and merge into a single EvolverResult."""
        insights: list[Insight] = []
        candidates: dict[str, list[PromptCandidate]] = {}
        paradigm_shift = False
        deprecation_targets: list[str] = []

        # Shared across mechanisms — computed once
        playbook = self._rebuild_playbook(harness_state)
        base_prompt = next(iter(harness_state.system_prompts.values()), None)

        # --- 1. Reflector: playbook insights ---
        # Pass the base system prompt so the Reflector can avoid
        # adding playbook entries that duplicate the static prompt.
        if self.reflector is not None and episodes:
            try:
                batch_sz = self.reflector.config.reflection_batch_size
                for i in range(0, len(episodes), batch_sz):
                    batch = episodes[i : i + batch_sz]
                    batch_insights = self.reflector.reflect(
                        batch, playbook, base_prompt=base_prompt,
                    )
                    # Auto-tag insights with source episode metadata for
                    # cleaner attribution when using per-sample reflection.
                    if batch_sz <= 2 and batch_insights:
                        ep_tags = self._extract_episode_tags(batch)
                        for insight in batch_insights:
                            existing = set(insight.tags) if insight.tags else set()
                            insight.tags = list(existing | ep_tags)
                    insights.extend(batch_insights)
            except Exception:
                log.exception("Reflector failed in LocalEvolver")

        # --- 2. GEPA: prompt mutation/crossover ---
        if self.prompt_evolver is not None and harness_state.pareto_fronts:
            try:
                gepa_candidates = self._run_gepa(episodes, harness_state)
                for bench, cands in gepa_candidates.items():
                    candidates.setdefault(bench, []).extend(cands)
            except Exception:
                log.exception("GEPA failed in LocalEvolver")

        # --- 3. Paradigm breakthrough on stagnation ---
        if self.paradigm is not None and context.is_stagnating:
            try:
                paradigm_insights = self.paradigm.generate(
                    playbook=playbook,
                    reward_history=context.reward_history,
                    tried_paradigms=context.tried_paradigms,
                )
                if paradigm_insights:
                    paradigm_shift = True
                    insights.extend(paradigm_insights)
                    # Deprecate non-paradigm entries
                    for entry_dict in harness_state.playbook_entries:
                        tags = entry_dict.get("tags", [])
                        if "paradigm" not in tags:
                            entry_id = entry_dict.get("id", "")
                            if entry_id:
                                deprecation_targets.append(entry_id)
            except Exception:
                log.exception("Paradigm breakthrough failed in LocalEvolver")

        return EvolverResult(
            insights=insights,
            candidates=candidates,
            paradigm_shift=paradigm_shift,
            deprecation_targets=deprecation_targets,
            run_id="",
            provenance=Provenance(backend="local"),
        )

    @staticmethod
    def _extract_episode_tags(episodes: list[Episode]) -> set[str]:
        """Extract category/bench tags from episodes for insight tagging."""
        tags: set[str] = set()
        for ep in episodes:
            if getattr(ep, "bench", None):
                tags.add(ep.bench)
            meta = getattr(ep, "metadata", None) or {}
            for key in ("entropic_category", "car_category", "task_category"):
                val = meta.get(key)
                if val:
                    tags.add(val)
        return tags

    @staticmethod
    def _candidate_from_dict(d: dict) -> PromptCandidate:
        """Reconstruct a PromptCandidate from a snapshot dict."""
        return PromptCandidate(
            id=d.get("id", PromptCandidate.new_id()),
            text=d.get("text", ""),
            per_task_scores=d.get("per_task_scores", {}),
            generation=d.get("generation", 0),
            parent_id=d.get("parent_id"),
        )

    def _rebuild_playbook(self, state: HarnessSnapshot) -> Playbook:
        """Reconstruct a Playbook from snapshot entries for sub-components."""
        from clawloop.learning_layers.harness import PlaybookEntry

        entries = []
        for e in state.playbook_entries:
            entries.append(PlaybookEntry(
                id=e.get("id", ""),
                content=e.get("content", ""),
                helpful=e.get("helpful", 0),
                harmful=e.get("harmful", 0),
                tags=e.get("tags", []),
                name=e.get("name", ""),
                description=e.get("description", ""),
                anti_patterns=e.get("anti_patterns", ""),
                category=e.get("category", "general"),
                superseded_by=e.get("superseded_by"),
            ))
        return Playbook(entries=entries)

    def _run_gepa(
        self,
        episodes: list[Episode],
        state: HarnessSnapshot,
    ) -> dict[str, list[PromptCandidate]]:
        """Run GEPA mutation and crossover across benches.

        Passes the rendered playbook to GEPA so it can see which strategies
        are already in the dynamic part and avoid duplicating them in the
        static system prompt.
        """
        result: dict[str, list[PromptCandidate]] = {}
        pe = self.prompt_evolver

        # Render playbook for GEPA context
        playbook = self._rebuild_playbook(state)
        playbook_text = playbook.render() or None

        for bench, front_data in state.pareto_fronts.items():
            if not front_data:
                continue

            best_dict = max(
                front_data,
                key=lambda c: (
                    sum(c.get("per_task_scores", {}).values()) / max(len(c.get("per_task_scores", {})), 1)
                ),
            )
            best = self._candidate_from_dict(best_dict)

            bench_candidates: list[PromptCandidate] = []

            # Mutation from failure episodes
            bench_failures = [
                ep for ep in episodes
                if ep.bench == bench and ep.summary.effective_reward() < 0
            ]
            if bench_failures:
                for _ in range(pe.config.max_mutations_per_step):
                    try:
                        child = pe.mutate(best, bench_failures, playbook_context=playbook_text)
                        if child is not None:
                            bench_candidates.append(child)
                    except Exception:
                        log.exception("Mutation failed for bench %s", bench)
                        break

            # Crossover
            if len(front_data) >= 2 and pe.config.max_crossovers_per_step > 0:
                a = self._candidate_from_dict(front_data[0])
                b = self._candidate_from_dict(front_data[1])
                for _ in range(pe.config.max_crossovers_per_step):
                    try:
                        child = pe.crossover(a, b, playbook_context=playbook_text)
                        if child is not None:
                            bench_candidates.append(child)
                    except Exception:
                        log.exception("Crossover failed for bench %s", bench)
                        break

            if bench_candidates:
                result[bench] = bench_candidates

        return result
