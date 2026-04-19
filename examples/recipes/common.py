"""Shared helpers for ClawLoop benchmark recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawloop.harness_backends.local import LocalEvolver


def build_local_evolver(
    reflector_model: str | None,
    api_key: str = "",
    api_base: str | None = None,
    reflection_batch_size: int = 1,
) -> LocalEvolver:
    """Build a LocalEvolver with an optional Reflector.

    Use this in every recipe instead of constructing Reflector and LocalEvolver
    inline, to ensure the wiring stays correct as the API evolves.
    """
    from clawloop.harness_backends.local import LocalEvolver

    reflector = None
    if reflector_model:
        from clawloop.core.reflector import Reflector, ReflectorConfig
        from clawloop.llm import LiteLLMClient

        reflector = Reflector(
            client=LiteLLMClient(
                model=reflector_model,
                api_key=api_key,
                api_base=api_base,
            ),
            config=ReflectorConfig(reflection_batch_size=reflection_batch_size),
        )

    return LocalEvolver(reflector=reflector)
