"""Serialization helpers for SkyDiscover <-> ClawLoop harness programs.

Integrates with SkyDiscover (https://github.com/skydiscover-ai/skydiscover)
by UC Berkeley Sky Computing Lab — Apache 2.0. No SkyDiscover code is copied.

Converts between HarnessSnapshot (ClawLoop's internal state) and the JSON
"program" files that SkyDiscover evolves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clawloop.core.evolver import EvolverResult, HarnessSnapshot
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
        e: dict[str, Any] = {
            "content": entry.get("content", ""),
            "tags": entry.get("tags", []),
            "helpful": entry.get("helpful", 0),
            "harmful": entry.get("harmful", 0),
        }
        # Preserve ID for deterministic diff matching with duplicates
        if entry.get("id"):
            e["id"] = entry["id"]
        playbook.append(e)

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
    raw = json.loads(Path(program_path).read_text())

    # SkyDiscover LLM mutations may produce valid JSON that isn't a dict
    # (e.g., a list or a quoted string). Normalize to our expected schema.
    if isinstance(raw, dict):
        program = raw
    elif isinstance(raw, str):
        program = {"system_prompt": raw, "playbook": [], "model": ""}
    else:
        program = {"system_prompt": str(raw), "playbook": [], "model": ""}

    insights: list[Insight] = []
    candidates: dict[str, list[PromptCandidate]] = {}

    # --- Diff playbook ---
    # Normalize: guard against non-list playbook (LLM may produce null/string)
    raw_playbook = program.get("playbook", [])
    evolved_playbook: list[dict[str, Any]] = [
        e for e in (raw_playbook if isinstance(raw_playbook, list) else [])
        if isinstance(e, dict)
    ]

    # Build lookup structures from original entries.
    # Prefer ID-based matching (deterministic with duplicates), fall back
    # to content-based matching for entries without IDs.
    original_by_id: dict[str, dict[str, Any]] = {}
    original_by_content: dict[str, list[dict[str, Any]]] = {}
    for e in original.playbook_entries:
        eid = e.get("id", "")
        if eid:
            original_by_id[eid] = e
        c = e.get("content", "")
        original_by_content.setdefault(c, []).append(e)

    matched_ids: set[str] = set()
    matched_content_counts: dict[str, int] = {}  # content -> count matched

    for entry in evolved_playbook:
        content = entry.get("content", "")
        eid = entry.get("id", "")

        # Try ID match first (deterministic)
        if eid and eid in original_by_id and eid not in matched_ids:
            matched_ids.add(eid)
            # Tag-only changes suppressed (Harness limitation — see comment above)
            continue

        # Fall back to content match
        orig_list = original_by_content.get(content, [])
        match_idx = matched_content_counts.get(content, 0)

        # Skip entries already matched by ID
        while match_idx < len(orig_list) and orig_list[match_idx].get("id", "") in matched_ids:
            match_idx += 1

        if match_idx < len(orig_list):
            orig = orig_list[match_idx]
            matched_ids.add(orig.get("id", ""))
            matched_content_counts[content] = match_idx + 1
            # Tag-only changes suppressed (Harness Curator increments helpful
            # and resets embeddings on "update" but does NOT update tags)
        else:
            # New entry added by evolution
            matched_content_counts[content] = match_idx + 1
            insights.append(Insight(
                content=content,
                tags=entry.get("tags", []),
                action="add",
            ))

    # Entries removed by evolution: any original entries not matched
    for e in original.playbook_entries:
        eid = e.get("id", "")
        if eid and eid not in matched_ids:
            insights.append(Insight(
                content=e.get("content", ""),
                action="remove",
                target_entry_id=eid or None,
            ))
        elif not eid:
            # No ID — check content-based matching
            content = e.get("content", "")
            matched_count = matched_content_counts.get(content, 0)
            total_with_content = len(original_by_content.get(content, []))
            if matched_count < total_with_content:
                matched_content_counts[content] = matched_count + 1
            else:
                insights.append(Insight(
                    content=content,
                    action="remove",
                    target_entry_id=None,
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
        # Provenance is set by the caller (SkyDiscoverAdaEvolve.evolve)
        # with actual runtime metadata from the DiscoveryResult.
    )
