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
from clawloop.layers.harness import Insight, Playbook, PromptCandidate

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

        # --- 1. Reflector: playbook insights ---
        if self.reflector is not None and episodes:
            try:
                playbook = self._rebuild_playbook(harness_state)
                batch_sz = self.reflector.config.reflection_batch_size
                for i in range(0, len(episodes), batch_sz):
                    batch = episodes[i : i + batch_sz]
                    batch_insights = self.reflector.reflect(batch, playbook)
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
                playbook = self._rebuild_playbook(harness_state)
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

    def _rebuild_playbook(self, state: HarnessSnapshot) -> Playbook:
        """Reconstruct a Playbook from snapshot entries for sub-components."""
        from clawloop.layers.harness import PlaybookEntry

        entries = []
        for e in state.playbook_entries:
            entries.append(PlaybookEntry(
                id=e.get("id", ""),
                content=e.get("content", ""),
                helpful=e.get("helpful", 0),
                harmful=e.get("harmful", 0),
                tags=e.get("tags", []),
            ))
        return Playbook(entries=entries)

    def _run_gepa(
        self,
        episodes: list[Episode],
        state: HarnessSnapshot,
    ) -> dict[str, list[PromptCandidate]]:
        """Run GEPA mutation and crossover across benches."""
        result: dict[str, list[PromptCandidate]] = {}
        pe = self.prompt_evolver

        for bench, front_data in state.pareto_fronts.items():
            if not front_data:
                continue

            # Reconstruct best candidate from snapshot
            best_dict = max(
                front_data,
                key=lambda c: (
                    sum(c.get("scores", {}).values()) / max(len(c.get("scores", {})), 1)
                ),
            )
            best = PromptCandidate(
                id=best_dict.get("id", PromptCandidate.new_id()),
                text=best_dict.get("text", ""),
                per_task_scores=best_dict.get("scores", {}),
            )

            bench_candidates: list[PromptCandidate] = []

            # Mutation from failure episodes
            bench_failures = [
                ep for ep in episodes
                if ep.bench == bench and ep.summary.effective_reward() < 0
            ]
            if bench_failures:
                for _ in range(pe.config.max_mutations_per_step):
                    try:
                        child = pe.mutate(best, bench_failures)
                        if child is not None:
                            bench_candidates.append(child)
                    except Exception:
                        log.exception("Mutation failed for bench %s", bench)
                        break

            # Crossover
            if len(front_data) >= 2 and pe.config.max_crossovers_per_step > 0:
                a_dict, b_dict = front_data[0], front_data[1]
                a = PromptCandidate(
                    id=a_dict.get("id", PromptCandidate.new_id()),
                    text=a_dict.get("text", ""),
                    per_task_scores=a_dict.get("scores", {}),
                )
                b = PromptCandidate(
                    id=b_dict.get("id", PromptCandidate.new_id()),
                    text=b_dict.get("text", ""),
                    per_task_scores=b_dict.get("scores", {}),
                )
                for _ in range(pe.config.max_crossovers_per_step):
                    try:
                        child = pe.crossover(a, b)
                        if child is not None:
                            bench_candidates.append(child)
                    except Exception:
                        log.exception("Crossover failed for bench %s", bench)
                        break

            if bench_candidates:
                result[bench] = bench_candidates

        return result
