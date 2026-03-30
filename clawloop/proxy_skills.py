"""Inject and strip ClawLoop playbook skills in OpenAI message lists.

Pure functions — no side-effects, no dependencies beyond stdlib.
The sentinel is an HTML comment so it is invisible to models that render
markdown.
"""

from __future__ import annotations

SENTINEL = "<!-- clawloop-skills:v1 -->"


def inject_skills(messages: list[dict], skills_text: str) -> list[dict]:
    """Prepend a system message carrying *skills_text* guarded by SENTINEL.

    * If *skills_text* is empty/whitespace, return *messages* unchanged.
    * Any pre-existing message that contains the sentinel is removed first
      (idempotent on retry).
    * The original list is never mutated.
    """
    if not skills_text or not skills_text.strip():
        return messages

    cleaned = [m for m in messages if SENTINEL not in m.get("content", "")]
    skills_msg: dict = {"role": "system", "content": f"{SENTINEL}\n{skills_text}"}
    return [skills_msg, *cleaned]


def strip_skills(messages: list[dict]) -> list[dict]:
    """Remove any message whose content contains the SENTINEL.

    The original list is never mutated.
    """
    return [m for m in messages if SENTINEL not in m.get("content", "")]
