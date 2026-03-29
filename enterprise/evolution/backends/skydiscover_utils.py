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
    # Use (content, index) as key to handle duplicate content strings safely.
    # Build a multimap: content -> list of original entries (preserving dupes).
    original_by_content: dict[str, list[dict[str, Any]]] = {}
    for e in original.playbook_entries:
        c = e.get("content", "")
        original_by_content.setdefault(c, []).append(e)

    evolved_playbook: list[dict[str, Any]] = program.get("playbook", [])

    # Track which original entries have been matched (by content + index)
    matched_originals: dict[str, int] = {}  # content -> count matched so far
    for entry in evolved_playbook:
        content = entry.get("content", "")
        orig_list = original_by_content.get(content, [])
        match_idx = matched_originals.get(content, 0)

        if match_idx < len(orig_list):
            # Matched an existing entry
            orig = orig_list[match_idx]
            matched_originals[content] = match_idx + 1
            # Tag-only changes are suppressed. The Harness Curator applies
            # "update" insights by incrementing helpful and resetting
            # embeddings — but does NOT update tags. Emitting the insight
            # would inflate helpful scores and churn embeddings for no
            # functional effect. Suppressed until Harness supports tag
            # updates (upstream limitation, not enterprise).
        else:
            # New entry added by evolution
            matched_originals[content] = match_idx + 1
            insights.append(Insight(
                content=content,
                tags=entry.get("tags", []),
                action="add",
            ))

    # Entries removed by evolution: any original entries not matched
    for content, orig_list in original_by_content.items():
        matched_count = matched_originals.get(content, 0)
        for orig in orig_list[matched_count:]:
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
        # Provenance is set by the caller (SkyDiscoverAdaEvolve.evolve)
        # with actual runtime metadata from the DiscoveryResult.
    )
