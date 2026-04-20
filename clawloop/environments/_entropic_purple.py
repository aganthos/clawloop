# clawloop/environments/_entropic_purple.py
"""A2A purple agent server for Entropic CRMArenaPro with clawloop harness injection."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar
from uuid import uuid4

from clawloop.environments._purple_base import (
    _PurpleAgentBase,
    create_app,
    start_purple_server,
)

log = logging.getLogger(__name__)

__all__ = ["EntropicPurpleAgent", "create_app", "start_purple_server"]


class EntropicPurpleAgent(_PurpleAgentBase):
    """Purple agent for Entropic CRMArenaPro: formats CRM task, flat message envelope."""

    default_bench: ClassVar[str] = "entropic"
    agent_card_name: ClassVar[str] = "clawloop-entropic-purple-agent"
    agent_card_description: ClassVar[str] = "ClawLoop harness-optimized CRM agent under test"
    agent_card_skills: ClassVar[list[dict]] = [
        {
            "id": "crm_assistant",
            "name": "CRM Assistant",
            "description": "Agent under test for Entropic CRMArenaPro evaluation",
            "tags": ["benchmark", "entropic", "crmarena"],
        }
    ]

    @staticmethod
    def _format_crm_task(raw_text: str) -> str:
        """Parse CRM task JSON and format as a readable prompt.

        The green agent sends ``json.dumps(task_context)`` as an A2A TextPart.
        We extract the structured fields and present them clearly to the LLM.
        If the text isn't valid JSON (or lacks ``prompt``), return it unchanged.
        """
        try:
            ctx = json.loads(raw_text)
        except (ValueError, TypeError):
            return raw_text

        if not isinstance(ctx, dict) or "prompt" not in ctx:
            return raw_text

        parts: list[str] = []

        persona = ctx.get("persona", "")
        if persona:
            parts.append(f"Persona: {persona}")

        parts.append(f"\nTask: {ctx['prompt']}")

        required = ctx.get("required_context", "")
        if required and required.strip():
            parts.append(f"\nContext:\n{required}")

        entropy = ctx.get("entropy")
        if entropy:
            parts.append(
                f"\nNote: Column names may have been modified "
                f"(drift_level={entropy.get('drift_level', '?')}). "
                "Adapt to any schema changes in the context."
            )

        parts.append(
            "\nProvide a direct, concise answer. "
            "If the task asks for IDs or specific values, return only those values."
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_task_tags(raw_text: str) -> set[str] | None:
        """Extract task category from CRM task JSON for selective playbook retrieval."""
        try:
            ctx = json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(ctx, dict):
            return None
        tags: set[str] = set()
        cat = ctx.get("task_category")
        if cat:
            tags.add(cat)
        return tags or None

    def _build_initial_messages(self, text_parts: list[str]) -> list[dict]:
        raw_text = "\n".join(text_parts)
        task_tags = self._extract_task_tags(raw_text)
        harness_prompt = self.harness.system_prompt(self.bench, task_tags=task_tags)
        system_content = harness_prompt or "You are a helpful CRM assistant."

        out: list[dict] = [{"role": "system", "content": system_content}]
        user_text = self._format_crm_task(raw_text)
        if user_text.strip():
            out.append({"role": "user", "content": user_text})
        return out

    def _format_a2a_response(self, assistant_msg: Any) -> dict:
        # Return Message directly (not wrapped) — a2a-sdk expects result=Message
        return {
            "kind": "message",
            "messageId": uuid4().hex,
            "role": "agent",
            "parts": self._build_message_parts(assistant_msg),
        }
