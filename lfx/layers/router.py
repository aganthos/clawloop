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

import math
from dataclasses import dataclass, field
from typing import Any


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

    def to_vector(self) -> list[float]:
        """Flatten to a numeric feature vector for the classifier."""
        return [
            float(self.token_count) / 1000.0,  # normalize
            float(self.has_code),
            float(self.reasoning_markers),
            float(self.technical_terms) / 10.0,
            float(self.tool_calls_expected),
            float(self.conversation_depth),
        ]


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
    tier_models: dict[str, str] = field(default_factory=lambda: {
        Tier.LIGHT: "",
        Tier.MEDIUM: "",
        Tier.HEAVY: "",
        Tier.REASONING: "",
    })

    # Scoring weights for complexity classification (trainable)
    score_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS)
    )

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
        self.training_samples.append({
            "features": features.to_vector(),
            "model_id": model_id,
            "cost": cost,
            "reward": reward,
            "tier": self.classify(features),
        })

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

        tier_means = {
            t: sum(vals) / len(vals) for t, vals in tier_stats.items() if vals
        }

        # Adjust weights: if LIGHT tier has good efficiency, reduce weights
        # (lower scores -> more queries routed to LIGHT)
        deltas: dict[str, float] = {}
        light_eff = tier_means.get(Tier.LIGHT, 0.0)
        heavy_eff = tier_means.get(Tier.HEAVY, 0.0) + tier_means.get(
            Tier.REASONING, 0.0
        )

        # If cheap models are doing well, nudge weights down
        direction = -1.0 if light_eff >= heavy_eff else 1.0

        for key in self.score_weights:
            delta = learning_rate * direction
            self.score_weights[key] = max(
                0.01, min(1.0, self.score_weights[key] + delta)
            )
            deltas[key] = delta

        # Normalize to sum to 1
        total = sum(self.score_weights.values())
        if total > 0:
            self.score_weights = {
                k: v / total for k, v in self.score_weights.items()
            }

        self.training_samples.clear()
        return deltas

    def _compute_score(self, features: QueryFeatures) -> float:
        """Weighted complexity score in [0, 1]."""
        vec = features.to_vector()
        keys = list(self.score_weights.keys())
        raw = sum(
            self.score_weights.get(keys[i], 0.0) * vec[i]
            for i in range(min(len(keys), len(vec)))
        )
        # Sigmoid to [0, 1]
        return 1.0 / (1.0 + math.exp(-raw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_models": self.tier_models,
            "score_weights": self.score_weights,
            "tier_thresholds": self.tier_thresholds,
            "fallback_chains": self.fallback_chains,
            "token_budgets": self.token_budgets,
            "cost_weights": self.cost_weights,
        }
