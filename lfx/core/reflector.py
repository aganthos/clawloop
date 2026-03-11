"""Reflector — LLM-based trace analysis for the ACE learning loop.

The Reflector reads episode traces and the current playbook, then asks an LLM
to extract general strategies (not task-specific answers).  Its output is a
list of Insight objects that the Curator (in Harness) applies as playbook
deltas.

Inspired by ACE's Reflector with SkyDiscover's sibling context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from lfx.core.episode import Episode
from lfx.layers.harness import Insight, Playbook

log = logging.getLogger(__name__)

# Maximum number of characters to include from each message in episode traces.
_MSG_TRUNCATE_LEN = 500

# Maximum number of insights the LLM is asked to produce per batch.
_MAX_INSIGHTS_PER_BATCH = 3

_SYSTEM_PROMPT = """\
You are a Reflector in a meta-learning system.  Your job is to analyze agent
episode traces and the current playbook, then extract **general strategies**
that will help the agent perform better in the future.

Rules:
- Extract reusable strategies, NOT task-specific answers.
- Each insight should describe a pattern, heuristic, or failure mode.
- Do not repeat entries already in the playbook unless you are updating them.
- Return a JSON array of insight objects (max {max_insights} per batch).

Each insight object must have these fields:
  - "action": one of "add", "update", "remove"
  - "content": a concise strategy description (string)
  - "target_entry_id": playbook entry ID to update/remove (string or null for "add")
  - "tags": list of short category tags (list of strings)
  - "source_episode_ids": list of episode IDs that motivated this insight

Return ONLY the JSON array.  No explanation, no markdown outside the JSON.
""".strip()


def _sanitize_str(text: str) -> str:
    """Strip null bytes from a string (defense-in-depth)."""
    return text.replace("\x00", "")


def _sanitize_obj(obj: Any) -> Any:
    """Recursively sanitize all string values in a JSON-like structure."""
    if isinstance(obj, str):
        return _sanitize_str(obj)
    if isinstance(obj, dict):
        return {_sanitize_str(k) if isinstance(k, str) else k: _sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_obj(item) for item in obj]
    return obj


@dataclass
class ReflectorConfig:
    """Configuration for the Reflector."""

    temperature: float = 0.7
    max_episodes_per_prompt: int = 5
    max_tokens: int = 2000


@dataclass
class Reflector:
    """LLM-based trace analyzer that produces Insight objects.

    The Reflector examines recent episode traces together with the current
    playbook and (optionally) sibling context to produce actionable insights
    for the Curator.
    """

    client: Any  # LLMClient protocol
    config: ReflectorConfig = field(default_factory=ReflectorConfig)

    def reflect(
        self,
        episodes: list[Episode],
        playbook: Playbook,
        *,
        sibling_context: str | None = None,
    ) -> list[Insight]:
        """Analyze episodes and return a list of Insight objects.

        Returns [] if *episodes* is empty (no LLM call is made).
        Returns [] on parse failure (logs a warning, does not raise).
        """
        if not episodes:
            return []

        # Limit episodes to max_episodes_per_prompt
        episodes = episodes[: self.config.max_episodes_per_prompt]

        user_prompt = self._build_prompt(episodes, playbook, sibling_context)
        system_prompt = _SYSTEM_PROMPT.format(max_insights=_MAX_INSIGHTS_PER_BATCH)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.complete(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return self._parse_response(str(response), episodes)

    def _build_prompt(
        self,
        episodes: list[Episode],
        playbook: Playbook,
        sibling_context: str | None,
    ) -> str:
        """Assemble the user prompt from playbook, episode traces, and sibling context."""
        sections: list[str] = []

        # -- CURRENT PLAYBOOK --
        pb_text = playbook.render()
        if pb_text:
            sections.append(f"## CURRENT PLAYBOOK\n{_sanitize_str(pb_text)}")
        else:
            sections.append("## CURRENT PLAYBOOK\n(empty)")

        # -- EPISODE TRACES (JSON-structured) --
        trace_objects: list[dict[str, Any]] = []
        for ep in episodes:
            trace_messages: list[dict[str, str]] = []
            for msg in ep.messages:
                content = _sanitize_str(msg.content)
                if len(content) > _MSG_TRUNCATE_LEN:
                    content = content[:_MSG_TRUNCATE_LEN] + "..."
                trace_messages.append({
                    "role": _sanitize_str(msg.role),
                    "content": content,
                })
            trace_objects.append(_sanitize_obj({
                "id": ep.id,
                "task_id": ep.task_id,
                "bench": ep.bench,
                "reward": ep.summary.total_reward,
                "messages": trace_messages,
            }))

        trace_json = json.dumps(trace_objects, indent=2)
        sections.append(f"## EPISODE TRACES\n```json\n{trace_json}\n```")

        # -- SIBLING CONTEXT --
        if sibling_context:
            sections.append(f"## SIBLING CONTEXT\n{_sanitize_str(sibling_context)}")

        return "\n\n".join(sections)

    def _parse_response(
        self,
        response: str,
        episodes: list[Episode],
    ) -> list[Insight]:
        """Extract JSON from the LLM response and create Insight objects.

        Handles ```json``` fenced code blocks.  Returns [] on any parse failure.
        """
        text = response.strip()

        # Strip ```json ... ``` fencing if present
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Reflector: failed to parse LLM response as JSON: %s", exc)
            return []

        if not isinstance(data, list):
            log.warning("Reflector: expected JSON array, got %s", type(data).__name__)
            return []

        episode_ids = [ep.id for ep in episodes]
        insights: list[Insight] = []

        for item in data:
            if not isinstance(item, dict):
                log.warning("Reflector: skipping non-dict item in response array, got %s", type(item).__name__)
                continue
            try:
                insight = Insight(
                    action=item.get("action", "add"),
                    content=item.get("content", ""),
                    target_entry_id=item.get("target_entry_id"),
                    tags=item.get("tags", []),
                    source_episode_ids=item.get("source_episode_ids", episode_ids),
                )
                insights.append(insight)
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("Reflector: skipping malformed insight: %s", exc)
                continue

        return insights
