"""Router layer — trainable model routing classifier.

Maps incoming queries to the optimal model based on complexity, cost, and
historical reward.  Aligned with RouteLLM/LLMRouter/ClawRouter:

  1. **Complexity scoring**: Multi-dimension weighted scorer classifies each
     query into a tier (LIGHT / MEDIUM / HEAVY / REASONING).
  2. **Tier -> model mapping**: Each tier maps to a model ID.  Fallback chains
     provide graceful degradation.
  3. **Budget constraints**: Token and cost budgets gate model selection.
  4. **RL training**: The classifier is trainable from
     ``(task_embedding, model_id, cost, reward)`` tuples, learning which model
     gives the best reward/cost tradeoff per query type.
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

# -- Complexity tiers --


class Tier:
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    REASONING = "reasoning"

    ALL = [LIGHT, MEDIUM, HEAVY, REASONING]


# -- Routing features --


@dataclass
class QueryFeatures:
    """Features extracted from a query for complexity scoring.

    Based on ClawRouter's multi-dimension approach: token count, code markers,
    reasoning markers, technical density, etc.
    """

    token_count: int = 0
    has_code: bool = False
    reasoning_markers: int = 0  # "prove", "explain why", "step by step", ...
    technical_terms: int = 0
    tool_calls_expected: int = 0
    conversation_depth: int = 0  # number of prior turns

    # Canonical key order — must match DEFAULT_SCORE_WEIGHTS keys.
    FEATURE_KEYS: tuple[str, ...] = (
        "token_count",
        "has_code",
        "reasoning_markers",
        "technical_terms",
        "tool_calls_expected",
        "conversation_depth",
    )

    def to_dict(self) -> dict[str, float]:
        """Named feature map for score computation."""
        return {
            "token_count": float(self.token_count) / 1000.0,
            "has_code": float(self.has_code),
            "reasoning_markers": float(self.reasoning_markers),
            "technical_terms": float(self.technical_terms) / 10.0,
            "tool_calls_expected": float(self.tool_calls_expected),
            "conversation_depth": float(self.conversation_depth),
        }

    def to_vector(self) -> list[float]:
        """Flatten to a numeric feature vector (canonical key order)."""
        d = self.to_dict()
        return [d[k] for k in self.FEATURE_KEYS]


# -- Scoring weights --


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "token_count": 0.15,
    "has_code": 0.15,
    "reasoning_markers": 0.25,
    "technical_terms": 0.15,
    "tool_calls_expected": 0.15,
    "conversation_depth": 0.15,
}

DEFAULT_TIER_THRESHOLDS: dict[str, float] = {
    Tier.LIGHT: 0.0,
    Tier.MEDIUM: 0.25,
    Tier.HEAVY: 0.50,
    Tier.REASONING: 0.75,
}


@dataclass
class _RouterPending:
    """Accumulator for forward_backward signals. Drained by optim_step.
    Stores (QueryFeatures, model_id, cost, reward) tuples."""

    samples: list[tuple[QueryFeatures, str, float, float]] = field(default_factory=list)


# -- Router layer --


@dataclass
class Router:
    """Trainable model routing classifier.

    Routes queries to the cheapest model that can handle them, using a
    multi-dimension complexity scorer.  The scoring weights are trainable
    from episode reward data.

    Learning mechanism:
      - Collect ``(query_features, model_used, cost, reward)`` from episodes
      - Adjust ``score_weights`` to maximize reward/cost ratio per tier
      - The Pareto-optimal tradeoff between cost and quality emerges from
        the training data
    """

    # Tier -> model ID mapping
    tier_models: dict[str, str] = field(
        default_factory=lambda: {
            Tier.LIGHT: "",
            Tier.MEDIUM: "",
            Tier.HEAVY: "",
            Tier.REASONING: "",
        }
    )

    # Scoring weights for complexity classification (trainable)
    score_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))

    # Tier thresholds (trainable)
    tier_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TIER_THRESHOLDS)
    )

    # Ordered fallback chain (if primary model unavailable)
    fallback_chains: list[str] = field(default_factory=list)

    # Budget constraints
    token_budgets: dict[str, int] = field(default_factory=dict)
    cost_weights: dict[str, float] = field(default_factory=dict)

    # Training history
    training_samples: list[dict[str, Any]] = field(default_factory=list)

    # Pending forward_backward accumulator (not part of observable state)
    _pending: _RouterPending = field(default_factory=_RouterPending)

    def classify(self, features: QueryFeatures) -> str:
        """Score a query and return the complexity tier."""
        score = self._compute_score(features)
        # Walk thresholds from highest to lowest
        for tier in reversed(Tier.ALL):
            if score >= self.tier_thresholds.get(tier, 0.0):
                return tier
        return Tier.LIGHT

    def route(self, features: QueryFeatures) -> str:
        """Route a query to a model ID based on complexity."""
        tier = self.classify(features)
        model = self.tier_models.get(tier, "")
        if model:
            return model
        # Fallback: try next higher tier
        tier_idx = Tier.ALL.index(tier) if tier in Tier.ALL else 0
        for i in range(tier_idx + 1, len(Tier.ALL)):
            model = self.tier_models.get(Tier.ALL[i], "")
            if model:
                return model
        # Last resort: fallback chain
        return self.fallback_chains[0] if self.fallback_chains else ""

    def record_outcome(
        self,
        features: QueryFeatures,
        model_id: str,
        cost: float,
        reward: float,
    ) -> None:
        """Record a routing outcome for future training."""
        self.training_samples.append(
            {
                "features": features.to_dict(),
                "model_id": model_id,
                "cost": cost,
                "reward": reward,
                "tier": self.classify(features),
            }
        )

    def update_weights(self, learning_rate: float = 0.01) -> dict[str, float]:
        """Update score_weights from training samples.

        Simple gradient-free approach: for each feature dimension, adjust
        weight based on whether higher-tier routing yielded better reward/cost.
        Production systems would use RL (Router-R1) or learned classifiers.

        Returns the weight deltas applied.
        """
        if len(self.training_samples) < 2:
            return {}

        # Group by tier, compute mean reward/cost per tier
        tier_stats: dict[str, list[float]] = {}
        for sample in self.training_samples:
            tier = sample["tier"]
            cost = max(sample["cost"], 1e-6)
            efficiency = sample["reward"] / cost
            tier_stats.setdefault(tier, []).append(efficiency)

        tier_means = {t: sum(vals) / len(vals) for t, vals in tier_stats.items() if vals}

        # Adjust weights: if LIGHT tier has good efficiency, reduce weights
        # (lower scores -> more queries routed to LIGHT)
        deltas: dict[str, float] = {}
        light_eff = tier_means.get(Tier.LIGHT, 0.0)
        heavy_eff = tier_means.get(Tier.HEAVY, 0.0) + tier_means.get(Tier.REASONING, 0.0)

        # If cheap models are doing well, nudge weights down
        direction = -1.0 if light_eff >= heavy_eff else 1.0

        for key in self.score_weights:
            delta = learning_rate * direction
            self.score_weights[key] = max(0.01, min(1.0, self.score_weights[key] + delta))
            deltas[key] = delta

        # Normalize to sum to 1
        total = sum(self.score_weights.values())
        if total > 0:
            self.score_weights = {k: v / total for k, v in self.score_weights.items()}

        self.training_samples.clear()
        return deltas

    def _compute_score(self, features: QueryFeatures) -> float:
        """Weighted complexity score in [0, 1].

        Feature values are already normalized by ``QueryFeatures.to_dict()``
        and weights sum to 1.0, so the dot product is naturally bounded.
        We clamp to [0, 1] to guard against extreme feature values.
        """
        feat = features.to_dict()
        raw = sum(
            self.score_weights.get(key, 0.0) * feat.get(key, 0.0)
            for key in QueryFeatures.FEATURE_KEYS
        )
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_models": self.tier_models,
            "score_weights": self.score_weights,
            "tier_thresholds": self.tier_thresholds,
            "fallback_chains": self.fallback_chains,
            "token_budgets": self.token_budgets,
            "cost_weights": self.cost_weights,
        }

    # -- Layer protocol methods --

    def clear_pending_state(self) -> None:
        """Reset the internal pending accumulator."""
        self._pending = _RouterPending()

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Extract routing signals from episodes and accumulate in _pending.

        MUST NOT mutate observable state (score_weights, training_samples,
        tier_thresholds, etc.).
        """
        for episode in data.episodes:
            # Find model_id from first assistant message with model set
            model_id = ""
            for msg in episode.messages:
                if msg.role == "assistant" and msg.model:
                    model_id = msg.model
                    break

            # Compute cost from token_usage in summary
            cost = 0.0
            if episode.summary.token_usage is not None:
                cost = float(episode.summary.token_usage.total_tokens)

            # Get reward from summary
            reward = episode.summary.total_reward

            # Build QueryFeatures with token_count from user messages
            token_count = sum(
                msg.token_count or len(msg.content.split())
                for msg in episode.messages
                if msg.role == "user"
            )
            features = QueryFeatures(token_count=token_count)

            self._pending.samples.append((features, model_id, cost, reward))

        return Future.immediate(FBResult(status="ok"))

    def optim_step(self) -> Future[OptimResult]:
        """Apply pending samples via record_outcome + update_weights.

        Uses snapshot-rollback: if update_weights fails, restore state.
        """
        if not self._pending.samples:
            return Future.immediate(OptimResult(status="ok", updates_applied=0))

        # Snapshot
        snapshot_training = list(self.training_samples)
        snapshot_weights = dict(self.score_weights)

        try:
            # Feed samples via record_outcome (adds tier key)
            for features, model_id, cost, reward in self._pending.samples:
                self.record_outcome(features, model_id, cost, reward)

            # Update weights
            deltas = self.update_weights()

            # Drain pending
            self._pending.samples.clear()

            return Future.immediate(OptimResult(status="ok", updates_applied=len(deltas)))
        except Exception:
            # Rollback
            self.training_samples = snapshot_training
            self.score_weights = snapshot_weights
            self._pending.samples.clear()
            return Future.immediate(OptimResult(status="error", updates_applied=0))

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Route a query to a model based on query features."""
        raw = ctx.query_features
        if isinstance(raw, QueryFeatures):
            features = raw
        else:
            # Reconstruct QueryFeatures from raw dict with int() casts
            features = QueryFeatures(
                token_count=int(raw.get("token_count", 0)),
                has_code=bool(raw.get("has_code", False)),
                reasoning_markers=int(raw.get("reasoning_markers", 0)),
                technical_terms=int(raw.get("technical_terms", 0)),
                tool_calls_expected=int(raw.get("tool_calls_expected", 0)),
                conversation_depth=int(raw.get("conversation_depth", 0)),
            )

        model_id = self.route(features)
        tier = self.classify(features)
        return Future.immediate(SampleResult(output=model_id, metadata={"tier": tier}))

    def save_state(self, name: str = "") -> Future[SaveResult]:
        """Save current state."""
        return Future.immediate(SaveResult(name=name, status="ok"))

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        """Restore state from a saved dict."""
        self.tier_models = state.get("tier_models", {})
        self.score_weights = state.get("score_weights", dict(DEFAULT_SCORE_WEIGHTS))
        self.tier_thresholds = state.get("tier_thresholds", dict(DEFAULT_TIER_THRESHOLDS))
        self.fallback_chains = state.get("fallback_chains", [])
        self.token_budgets = state.get("token_budgets", {})
        self.cost_weights = state.get("cost_weights", {})
        # Clear training state and pending
        self.training_samples.clear()
        self._pending = _RouterPending()
        return Future.immediate(LoadResult(status="ok"))
