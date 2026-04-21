"""Two-phase fb→optim→rollback layer transaction.

Extracted from ``learning_loop`` — this module owns the subtle
invariants of the per-iteration training protocol:

1. Build one ``Datum`` per layer name.
2. Run ``forward_backward`` on every active layer, clearing pending
   state on error/skipped results. The harness gets an intensity-gate
   short-circuit that records ``status="skipped"`` without calling fb.
3. Track paradigm shifts onto ``agent_state.tried_paradigms`` *before*
   optim drains ``_pending``.
4. Snapshot every layer that had ok fb. If snapshotting fails,
   ``clear_pending_state`` fires on every snapshotted layer and optim is
   skipped entirely.
5. Run ``optim_step`` on every snapshotted layer. On the first error or
   exception, roll back *every* snapshotted layer (even those whose
   optim hadn't run yet) to its pre-optim state.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from clawloop.core.episode import Episode
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.types import Datum, FBResult
from clawloop.learning_layers.harness import Harness

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)


@dataclass
class TransactionResult:
    """Outcome of one ``LayerTransaction.run()`` call."""

    fb_results: dict[str, FBResult]
    optim_failed: bool


class LayerTransaction:
    """One iteration's fb→optim→rollback protocol across active layers."""

    def __init__(
        self,
        layers: list[tuple[str, Any]],
        intensity: AdaptiveIntensity | None,
        episodes: list[Episode],
        agent_state: "AgentState",
    ) -> None:
        self._layers = layers
        self._intensity = intensity
        self._episodes = episodes
        self._agent_state = agent_state

    def run(self, iteration: int) -> TransactionResult:
        """Execute the transaction and return fb_results + optim_failed flag."""
        layer_datums: dict[str, Datum] = {
            "harness": Datum(episodes=self._episodes),
            "weights": Datum(episodes=self._episodes),
            "router": Datum(episodes=self._episodes),
        }
        fb_results = self._forward_backward(iteration, layer_datums)
        self._track_paradigm_shifts(fb_results)
        optim_failed = self._optim_with_rollback(fb_results)
        return TransactionResult(fb_results=fb_results, optim_failed=optim_failed)

    def _forward_backward(
        self,
        iteration: int,
        layer_datums: dict[str, Datum],
    ) -> dict[str, FBResult]:
        fb_results: dict[str, FBResult] = {}
        for name, layer in self._layers:
            if (
                name == "harness"
                and self._intensity is not None
                and not self._intensity.should_reflect(iteration)
            ):
                log.info("  skipping harness fb (adaptive intensity)")
                fb_results[name] = FBResult(status="skipped")
                continue
            if name in layer_datums:
                datum = layer_datums[name]
            else:
                log.warning("  unknown layer %s — using all episodes as fallback", name)
                datum = Datum(episodes=self._episodes)

            should_clear = False
            try:
                fb_result = layer.forward_backward(datum).result()
                fb_results[name] = fb_result
                if fb_result.status in ("error", "skipped"):
                    should_clear = True
            except Exception:
                log.exception("forward_backward failed for %s", name)
                fb_results[name] = FBResult(status="error")
                should_clear = True

            if should_clear:
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception("Failed to clear pending for %s", name)

        for name, result in fb_results.items():
            log.info("  fb %s: %s %s", name, result.status, result.metrics)
        return fb_results

    def _track_paradigm_shifts(self, fb_results: dict[str, FBResult]) -> None:
        """Append paradigm-tagged pending insights to ``tried_paradigms``.

        Must run before optim drains ``_pending`` — otherwise the insights
        we need to remember have already been consumed.
        """
        harness_fb = fb_results.get("harness")
        if (
            harness_fb is not None
            and harness_fb.metrics.get("paradigm_shifted")
            and isinstance(self._agent_state.harness, Harness)
        ):
            for insight in self._agent_state.harness.pending_paradigm_insights():
                self._agent_state.tried_paradigms.append(insight.content)

    def _optim_with_rollback(self, fb_results: dict[str, FBResult]) -> bool:
        layers_to_optim = [
            (name, layer)
            for name, layer in self._layers
            if fb_results.get(name, FBResult(status="error")).status not in ("error", "skipped")
        ]

        snapshots: dict[str, dict[str, Any]] = {}
        try:
            for name, layer in layers_to_optim:
                snapshots[name] = copy.deepcopy(layer.to_dict())
        except Exception:
            log.exception("Snapshot failed — skipping optim this iteration")
            for name, layer in layers_to_optim:
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception("Failed to clear pending for %s", name)
            return False

        optim_failed = False
        for name, layer in layers_to_optim:
            try:
                result = layer.optim_step().result()
                log.info(
                    "  optim %s: %s, %d updates",
                    name,
                    result.status,
                    result.updates_applied,
                )
                if result.status == "error":
                    optim_failed = True
                    log.error("  optim %s returned error — triggering rollback", name)
                    break
            except Exception:
                log.exception("optim_step failed for %s — triggering rollback", name)
                optim_failed = True
                break

        if optim_failed:
            log.warning("  rolling back all layers to pre-optim state")
            for name, layer in layers_to_optim:
                if name in snapshots:
                    try:
                        lr = layer.load_state(snapshots[name]).result()
                        if lr.status != "ok":
                            log.error("  rollback returned %s for %s", lr.status, name)
                    except Exception:
                        log.exception("  rollback failed for %s", name)

        return optim_failed
