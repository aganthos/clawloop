"""Serialization helpers for SkyDiscover <-> ClawLoop harness programs.

Converts between HarnessSnapshot (ClawLoop's internal state) and the JSON
"program" files that SkyDiscover evolves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clawloop.core.evolver import EvolverResult, HarnessSnapshot, Provenance
from clawloop.layers.harness import Insight, PromptCandidate


# ---------------------------------------------------------------------------
# Program JSON schema (what SkyDiscover sees as a "program"):
#
# {
#   "system_prompt": "...",
#   "playbook": [{"content": "...", "tags": [...], "helpful": N, "harmful": N}, ...],
#   "model": "..."
# }
# ---------------------------------------------------------------------------


def harness_to_program(snapshot: HarnessSnapshot, output_path: str) -> str:
    """Serialize a HarnessSnapshot to a JSON program file for SkyDiscover.

    Returns the absolute path of the written file.
    """
    # Pick the first system prompt (single-bench default)
    system_prompt = next(iter(snapshot.system_prompts.values()), "")

    playbook = []
    for entry in snapshot.playbook_entries:
        playbook.append({
            "content": entry.get("content", ""),
            "tags": entry.get("tags", []),
            "helpful": entry.get("helpful", 0),
            "harmful": entry.get("harmful", 0),
        })

    program: dict[str, Any] = {
        "system_prompt": system_prompt,
        "playbook": playbook,
        "model": "",  # filled by caller if needed
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(program, indent=2))
    return str(path.resolve())


def program_to_evolver_result(
    program_path: str,
    original: HarnessSnapshot,
) -> EvolverResult:
    """Parse an evolved program JSON and diff against the original snapshot.

    Produces an EvolverResult with:
    - Insights for playbook changes (add/update/remove)
    - PromptCandidates if the system_prompt changed
    - Provenance tagged with backend="skydiscover_adaevolve"
    """
    program = json.loads(Path(program_path).read_text())

    insights: list[Insight] = []
    candidates: dict[str, list[PromptCandidate]] = {}

    # --- Diff playbook ---
    original_entries = {
        e.get("content", ""): e for e in original.playbook_entries
    }
    evolved_playbook: list[dict[str, Any]] = program.get("playbook", [])

    evolved_contents: set[str] = set()
    for entry in evolved_playbook:
        content = entry.get("content", "")
        evolved_contents.add(content)

        if content not in original_entries:
            # New entry added by evolution
            insights.append(Insight(
                content=content,
                tags=entry.get("tags", []),
                action="add",
            ))
        else:
            # Existing entry — check if tags changed
            orig = original_entries[content]
            if entry.get("tags", []) != orig.get("tags", []):
                orig_id = orig.get("id", "")
                insights.append(Insight(
                    content=content,
                    tags=entry.get("tags", []),
                    action="update",
                    target_entry_id=orig_id or None,
                ))

    # Entries removed by evolution
    for content, orig in original_entries.items():
        if content not in evolved_contents:
            orig_id = orig.get("id", "")
            insights.append(Insight(
                content=content,
                action="remove",
                target_entry_id=orig_id or None,
            ))

    # --- Diff system prompt ---
    original_prompt = next(iter(original.system_prompts.values()), "")
    evolved_prompt = program.get("system_prompt", "")

    if evolved_prompt and evolved_prompt != original_prompt:
        # Surface as a candidate for the first bench (or "default")
        bench = next(iter(original.system_prompts.keys()), "default")
        candidates[bench] = [
            PromptCandidate(
                id=PromptCandidate.new_id(),
                text=evolved_prompt,
                generation=original.playbook_generation + 1,
            )
        ]

    return EvolverResult(
        insights=insights,
        candidates=candidates,
        provenance=Provenance(backend="skydiscover_adaevolve"),
    )
