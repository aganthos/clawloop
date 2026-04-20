# clawloop/environments/_car_purple.py
"""A2A purple agent server for CAR-bench with clawloop harness injection."""

from __future__ import annotations

import logging
from typing import Any, ClassVar
from uuid import uuid4

from clawloop.environments._purple_base import (
    _PurpleAgentBase,
    create_app,
    start_purple_server,
)
from clawloop.learning_layers.harness import Harness

log = logging.getLogger(__name__)

__all__ = ["CarPurpleAgent", "create_app", "start_purple_server"]


class CarPurpleAgent(_PurpleAgentBase):
    """Purple agent for CAR-bench: parses System/User prompt, wraps replies in ``message``."""

    default_bench: ClassVar[str] = "car"
    agent_card_name: ClassVar[str] = "clawloop-purple-agent"
    agent_card_description: ClassVar[str] = "ClawLoop harness-optimized agent under test"
    agent_card_skills: ClassVar[list[dict]] = [
        {
            "id": "car_assistant",
            "name": "In-Car Voice Assistant",
            "description": "Agent under test for CAR-bench evaluation",
            "tags": ["benchmark", "car-bench"],
        }
    ]

    def __init__(
        self,
        model: str,
        harness: Harness,
        bench: str = "car",
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model, harness, bench, api_base, api_key)
        self._captured: dict[str, list[dict]] = {}

    def clear_all_sessions(self) -> None:
        super().clear_all_sessions()
        self._captured.clear()

    @staticmethod
    def _parse_first_message(raw_text: str) -> tuple[str, str]:
        """Parse ``System: ...\\n\\nUser: ...`` format from green agent."""
        if "System:" in raw_text and "\n\nUser:" in raw_text:
            parts = raw_text.split("\n\nUser:", 1)
            system = parts[0].replace("System:", "", 1).strip()
            user = parts[1].strip()
            return system, user
        return "", raw_text

    def _build_initial_messages(self, text_parts: list[str]) -> list[dict]:
        raw_text = text_parts[0] if text_parts else ""
        system_prompt, user_text = self._parse_first_message(raw_text)

        harness_prompt = self.harness.system_prompt(self.bench)
        if harness_prompt:
            system_prompt = f"{harness_prompt}\n\n{system_prompt}"

        out: list[dict] = [{"role": "system", "content": system_prompt}]
        if user_text:
            out.append({"role": "user", "content": user_text})
        return out

    def _format_a2a_response(self, assistant_msg: Any) -> dict:
        return {
            "message": {
                "messageId": uuid4().hex,
                "role": "agent",
                "parts": self._build_message_parts(assistant_msg),
            }
        }

    def _capture_assistant(self, context_id: str, normalized: dict) -> None:
        self._captured.setdefault(context_id, []).append(normalized)
