"""Paradigm Breakthrough — stagnation escape via LLM-generated strategic shifts.

When the learning loop stagnates (reward stops improving), this module asks a
strong LLM to propose fundamentally new strategic directions.  These get added
to the playbook as entries tagged ``[paradigm]``.

Inspired by SkyDiscover's approach to escaping local optima: instead of tweaking
existing strategies, the model is prompted to suggest entirely new paradigms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from clawloop.learning_layers.harness import Insight, Playbook

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a strategic innovation advisor for an AI learning system.

Your job is to propose FUNDAMENTALLY NEW strategic directions — not incremental
tweaks or minor variations of existing approaches.  Think paradigm shifts:
entirely different ways of framing the problem, novel algorithmic families,
or unconventional reasoning strategies.

Rules:
- Propose 1 to 3 new paradigms per call.
- Each paradigm must be genuinely distinct from the current playbook and from
  previously tried paradigms.
- Do NOT propose small adjustments to existing strategies.
- Focus on bold, creative, high-potential directions.

Respond with a JSON array of objects, each with a "content" key describing
the new paradigm.  Example:

[
  {"content": "Use Monte-Carlo tree search to explore reasoning paths"},
  {"content": "Frame the task as a debate between two competing agents"}
]

Return ONLY the JSON array, no other text.
"""


@dataclass
class ParadigmConfig:
    """Configuration for paradigm breakthrough generation."""

    max_paradigms: int = 3
    temperature: float = 0.9
    max_tokens: int = 1500


@dataclass
class ParadigmBreakthrough:
    """Generate fundamentally new strategic directions when learning stagnates.

    Uses a strong LLM to propose paradigm-level shifts rather than incremental
    tweaks.  All generated insights are tagged with ``"paradigm"`` and have
    ``action="add"`` so they get appended to the playbook.
    """

    client: Any
    config: ParadigmConfig = field(default_factory=ParadigmConfig)

    def generate(
        self,
        playbook: Playbook,
        reward_history: list[float],
        tried_paradigms: list[str],
    ) -> list[Insight]:
        """Propose new strategic paradigms via LLM.

        Args:
            playbook: Current playbook (rendered for context).
            reward_history: Recent reward values showing stagnation.
            tried_paradigms: Previously attempted paradigm descriptions
                to avoid repeating.

        Returns:
            A list of :class:`Insight` objects tagged ``"paradigm"`` with
            ``action="add"``, or ``[]`` on parse failure.
        """
        user_prompt = self._build_user_prompt(playbook, reward_history, tried_paradigms)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = str(self.client.complete(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ))
        except Exception:
            log.exception("LLM call failed during paradigm generation")
            return []

        return self._parse_response(raw)

    def _build_user_prompt(
        self,
        playbook: Playbook,
        reward_history: list[float],
        tried_paradigms: list[str],
    ) -> str:
        """Assemble the user prompt with context for the LLM."""
        parts: list[str] = []

        # Current playbook
        rendered = playbook.render()
        if rendered:
            parts.append(f"## Current Playbook\n{rendered}")
        else:
            parts.append("## Current Playbook\n(empty)")

        # Reward history
        if reward_history:
            history_str = ", ".join(f"{r:.3f}" for r in reward_history)
            parts.append(f"## Recent Reward History\n{history_str}")

        # Previously tried paradigms
        if tried_paradigms:
            tried_str = "\n".join(f"- {p}" for p in tried_paradigms)
            parts.append(
                f"## Previously Tried Paradigms (DO NOT repeat these)\n{tried_str}"
            )

        parts.append(
            "Propose 1 to 3 fundamentally new strategic directions. "
            "Respond with a JSON array only."
        )

        return "\n\n".join(parts)

    def _parse_response(self, raw: str) -> list[Insight]:
        """Parse the LLM's JSON response into Insight objects."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("Failed to parse paradigm response as JSON: %s", raw[:200])
            return []

        if not isinstance(data, list):
            log.warning("Paradigm response is not a JSON array")
            return []

        insights: list[Insight] = []
        for item in data:
            if not isinstance(item, dict) or "content" not in item:
                continue
            content = str(item["content"]).strip()
            if not content:
                continue
            insights.append(
                Insight(
                    content=content,
                    tags=["paradigm"],
                    action="add",
                )
            )
            if len(insights) >= self.config.max_paradigms:
                break

        return insights
