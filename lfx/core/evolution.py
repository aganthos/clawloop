"""PromptEvolver — LLM-based GEPA mutation and crossover operators.

Produces new PromptCandidate objects from existing ones. Mutation targets
failures (support episodes), crossover combines strengths of two
non-dominated candidates.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from lfx.core.episode import Episode
from lfx.layers.harness import PromptCandidate

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, return raw JSON string."""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


@dataclass
class EvolverConfig:
    """Configuration for the prompt evolver."""

    max_mutations_per_step: int = 2
    max_crossovers_per_step: int = 1
    max_episode_context: int = 3  # episodes to include in mutation prompt


@dataclass
class PromptEvolver:
    """LLM-based GEPA mutation and crossover operators."""

    llm: Any  # LLMClient protocol
    config: EvolverConfig = field(default_factory=EvolverConfig)

    def mutate(
        self, parent: PromptCandidate, feedback: list[Episode],
    ) -> PromptCandidate | None:
        """ASI-guided mutation: propose a targeted fix based on failing episodes.

        Returns a new candidate with parent lineage, or None if the LLM
        response cannot be parsed.
        """
        episode_ctx = feedback[: self.config.max_episode_context]
        episode_summaries = []
        for ep in episode_ctx:
            msgs = [
                f"{m.role}: {m.content[:200]}" for m in ep.messages[:6]
            ]
            episode_summaries.append(
                f"Task {ep.task_id} (reward={ep.summary.effective_reward():.2f}):\n"
                + "\n".join(msgs)
            )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a prompt optimization expert. Given a system prompt "
                    "and episodes where it failed, propose a targeted modification "
                    "to improve performance. Return ONLY a JSON object with a "
                    '"revised_prompt" key containing the full improved prompt text.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Current System Prompt\n{parent.text}\n\n"
                    f"## Failing Episodes\n"
                    + "\n---\n".join(episode_summaries)
                    + "\n\nPropose a revised system prompt that addresses these failures."
                ),
            },
        ]

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(_extract_json(text))
            revised = parsed.get("revised_prompt", "")
            if not revised or not isinstance(revised, str):
                log.warning("Mutation response missing revised_prompt")
                return None
        except (json.JSONDecodeError, AttributeError, KeyError) as exc:
            log.warning("Failed to parse mutation response: %s", exc)
            return None

        return PromptCandidate(
            id=PromptCandidate.new_id(),
            text=revised,
            generation=parent.generation + 1,
            parent_id=parent.id,
        )

    def crossover(
        self, a: PromptCandidate, b: PromptCandidate,
    ) -> PromptCandidate | None:
        """Combine strengths of two non-dominated candidates.

        Returns a new candidate, or None if the LLM response cannot be parsed.
        """
        a_tasks = ", ".join(
            f"{k}: {v:.2f}" for k, v in sorted(a.per_task_scores.items())
        ) or "no scores yet"
        b_tasks = ", ".join(
            f"{k}: {v:.2f}" for k, v in sorted(b.per_task_scores.items())
        ) or "no scores yet"

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a prompt optimization expert. Given two system prompts "
                    "with their per-task performance scores, create a hybrid that "
                    "combines their strengths. Return ONLY a JSON object with a "
                    '"revised_prompt" key containing the full combined prompt text.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Candidate A (scores: {a_tasks})\n{a.text}\n\n"
                    f"## Candidate B (scores: {b_tasks})\n{b.text}\n\n"
                    "Create a hybrid prompt that combines the strengths of both."
                ),
            },
        ]

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(_extract_json(text))
            revised = parsed.get("revised_prompt", "")
            if not revised or not isinstance(revised, str):
                log.warning("Crossover response missing revised_prompt")
                return None
        except (json.JSONDecodeError, AttributeError, KeyError) as exc:
            log.warning("Failed to parse crossover response: %s", exc)
            return None

        return PromptCandidate(
            id=PromptCandidate.new_id(),
            text=revised,
            generation=max(a.generation, b.generation) + 1,
            parent_id=a.id,  # primary parent
        )
