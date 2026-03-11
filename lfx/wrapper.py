"""lfx.wrap() — SDK wrapper that intercepts LLM calls for learning."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message


class WrappedClient:
    """Drop-in LLMClient replacement that intercepts calls for learning."""

    def __init__(self, client: Any, collector: EpisodeCollector) -> None:
        self._client = client
        self._collector = collector

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        # Extract lfx-specific kwargs before forwarding to the client
        task_id = kwargs.pop("task_id", uuid4().hex)

        response = self._client.complete(messages, **kwargs)

        ep_messages = []
        for m in messages:
            ep_messages.append(Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                name=m.get("name"),
            ))
        ep_messages.append(Message(role="assistant", content=str(response)))

        # Stable session_id from first user message (deterministic across runs)
        session_id = ""
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                session_id = hashlib.sha256(content.encode()).hexdigest()[:16]
                break

        self._collector.ingest(
            ep_messages, task_id=task_id, session_id=session_id,
        )

        return response


def wrap(client: Any, collector: EpisodeCollector) -> WrappedClient:
    """Wrap an LLMClient with live-mode episode collection.

    Usage::

        wrapped = lfx.wrap(my_client, collector=collector)
        result = wrapped.complete(messages)  # works exactly like before
    """
    return WrappedClient(client, collector)
