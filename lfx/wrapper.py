"""lfx.wrap() — SDK wrapper that intercepts LLM calls for learning."""

from __future__ import annotations

from typing import Any

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message


class WrappedClient:
    """Drop-in LLMClient replacement that intercepts calls for learning."""

    def __init__(self, client: Any, collector: EpisodeCollector) -> None:
        self._client = client
        self._collector = collector

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        response = self._client.complete(messages, **kwargs)

        ep_messages = []
        for m in messages:
            ep_messages.append(Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                name=m.get("name"),
            ))
        ep_messages.append(Message(role="assistant", content=response))

        session_id = ""
        for m in messages:
            if m.get("role") == "user":
                session_id = str(hash(m.get("content", "")))
                break

        self._collector.ingest(ep_messages, session_id=session_id)

        return response


def wrap(client: Any, collector: EpisodeCollector) -> WrappedClient:
    """Wrap an LLMClient with live-mode episode collection.

    Usage::

        wrapped = lfx.wrap(my_client, collector=collector)
        result = wrapped.complete(messages)  # works exactly like before
    """
    return WrappedClient(client, collector)
