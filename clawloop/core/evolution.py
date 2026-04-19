"""PromptEvolver — LLM-based GEPA mutation and crossover operators.

Produces new PromptCandidate objects from existing ones. Mutation targets
failures (support episodes), crossover combines strengths of two
non-dominated candidates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from clawloop.core.episode import Episode
from clawloop.core.parse import extract_json
from clawloop.learning_layers.harness import PromptCandidate

log = logging.getLogger(__name__)


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
        self,
        parent: PromptCandidate,
        feedback: list[Episode],
        playbook_context: str | None = None,
    ) -> PromptCandidate | None:
        """ASI-guided mutation: propose a targeted fix based on failing episodes.

        When *playbook_context* is provided (rendered playbook text), the LLM
        sees it so it can complement rather than duplicate playbook strategies.
        The system prompt is the static part (all queries), the playbook is
        the dynamic part (query-specific) — they must not overlap.

        Returns a new candidate with parent lineage, or None if the LLM
        response cannot be parsed.
        """
        episode_ctx = feedback[: self.config.max_episode_context]
        episode_summaries = []
        for ep in episode_ctx:
            msgs = [f"{m.role}: {m.content[:200]}" for m in ep.messages[:6]]
            episode_summaries.append(
                f"Task {ep.task_id} (reward={ep.summary.effective_reward():.2f}):\n"
                + "\n".join(msgs)
            )

        user_parts = [
            f"## Current System Prompt (static, applies to all queries)\n{parent.text}",
        ]
        if playbook_context:
            user_parts.append(
                "## Current Playbook (dynamic, appended per-query — do NOT "
                f"duplicate these)\n{playbook_context}"
            )
        user_parts.append("## Failing Episodes\n" + "\n---\n".join(episode_summaries))
        user_parts.append(
            "Propose a revised system prompt that addresses these failures. "
            "Only modify the static system prompt — do not include playbook strategies."
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a prompt optimization expert. Given a system prompt "
                    "and episodes where it failed, propose a targeted modification "
                    "to improve performance. The system prompt is the STATIC part "
                    "that applies to all queries. A separate dynamic playbook "
                    "(shown for reference) handles per-query strategies — do NOT "
                    "duplicate those in the system prompt. Return ONLY a JSON "
                    'object with a "revised_prompt" key containing the full '
                    "improved system prompt text."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(user_parts),
            },
        ]

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(extract_json(text))
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
        self,
        a: PromptCandidate,
        b: PromptCandidate,
        playbook_context: str | None = None,
    ) -> PromptCandidate | None:
        """Combine strengths of two non-dominated candidates.

        Returns a new candidate, or None if the LLM response cannot be parsed.
        """
        a_tasks = (
            ", ".join(f"{k}: {v:.2f}" for k, v in sorted(a.per_task_scores.items()))
            or "no scores yet"
        )
        b_tasks = (
            ", ".join(f"{k}: {v:.2f}" for k, v in sorted(b.per_task_scores.items()))
            or "no scores yet"
        )

        user_parts = [
            f"## Candidate A (scores: {a_tasks})\n{a.text}",
            f"## Candidate B (scores: {b_tasks})\n{b.text}",
        ]
        if playbook_context:
            user_parts.append(
                "## Current Playbook (dynamic, appended per-query — do NOT "
                f"duplicate these)\n{playbook_context}"
            )
        user_parts.append(
            "Create a hybrid system prompt that combines the strengths of both. "
            "Do not include strategies already covered by the playbook."
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a prompt optimization expert. Given two system prompts "
                    "with their per-task performance scores, create a hybrid that "
                    "combines their strengths. A separate dynamic playbook "
                    "(shown for reference) handles per-query strategies — do NOT "
                    "duplicate those. Return ONLY a JSON object with a "
                    '"revised_prompt" key containing the full combined prompt text.'
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(user_parts),
            },
        ]

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(extract_json(text))
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
