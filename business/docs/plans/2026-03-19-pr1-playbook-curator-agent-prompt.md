# Agent Prompt: PR1 — Playbook Curator

## Your Task

Implement PR1 ("Playbook Health — Conflict, Redundancy, Attribution, Curation") from `docs/plans/2026-03-17-competitive-learnings-roadmap.md`. That file is your implementation spec — read it first, focus on the PR1 section (Tasks 1.1–1.12).

## Branch

Create and work on `feat/playbook-curator` off `main`.

## Project Context

**lfx** = "Learning from Experience" — a unified learning API for AI agents. Three layers:
- **Harness** (`layers/harness.py`) — prompt optimization via GEPA Pareto-front + ACE-style playbook memory
- **Router** (`layers/router.py`) — model selection
- **Weights** (`layers/weights.py`) — LoRA/GRPO training via SkyRL/Harbor

The playbook is a list of `PlaybookEntry` items (reusable strategies) that get injected into agent system prompts. Currently it has critical gaps: no conflict detection, no redundancy detection, broken attribution (all entries get same signal from every episode), `prune()` is never called, entries grow without bound.

## Key Files You'll Modify

| File | What it does |
|------|-------------|
| `layers/harness.py` | `PlaybookEntry`, `Playbook`, `Harness`, `Insight`, `ParetoFront` — the main target |
| `core/reflector.py` | LLM-based trace analyzer producing `Insight` objects |
| `core/paradigm.py` | `ParadigmBreakthrough` — stagnation escape |
| `core/loop.py` | Learning loop orchestrator (has a `tried_paradigms=[]` TODO on line 308) |
| `core/episode.py` | `Episode`, `EpisodeSummary` — need `scored_at_generation` field |
| `llm.py` | `LLMClient` protocol, `MockLLMClient` — use MockLLMClient in tests |

## New Files You'll Create

| File | Purpose |
|------|---------|
| `core/embeddings.py` | `EmbeddingProvider` protocol, `cosine_similarity()`, `find_similar()`, `MockEmbedding` |
| `core/curator.py` | `PlaybookCurator`, `CurationResult`, `CuratorMetrics`, `ConsolidationReport` |
| `tests/test_curator.py` | Retrieve-classify-revise tests with mock embeddings + mock LLM |
| `tests/test_attribution.py` | Entry-level attribution tests |
| `tests/test_decay.py` | `effective_score()` temporal decay tests |
| `tests/test_consolidation.py` | Clustering, merging, pruning, max-size cap |

## Architecture Principles

- **Heuristic-first classification**: `cosine_sim > 0.95` = identical (no LLM), `> 0.8` + contradiction keywords = conflicting (no LLM), `0.6–0.8` = LLM classifies. Below threshold = unrelated.
- **Never delete on first pass**: conflict resolution creates new entry, marks originals with `superseded_by`. Superseded entries hidden from `render()` but retained.
- **Fallback path**: if embedding/LLM unavailable, curator degrades to direct `add` with warning log. Never hard-fail.
- **Generation versioning**: `playbook_generation` increments on structural changes (insights applied). Episodes stamped with `scored_at_generation` are skipped when stale.

## Testing

```bash
# Run all tests
cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -x -v

# Run specific test file
python -m pytest tests/test_curator.py -v
```

Existing tests in `tests/test_harness_signals.py` and `tests/test_layer_protocol.py` MUST continue to pass. Use `MockLLMClient` and `MockEmbedding` for all tests — no live API calls.

## Commit Style

- Format: `fix:`, `feat:`, or `chore:` + one line description
- NO Co-Authored-By lines in commits
- NO multi-line bodies
- Small, focused commits — one per task or logical unit

## Execution Order

Follow the task numbers in the roadmap (1.1 → 1.12). Each task builds on the previous. Write tests alongside implementation, not all at the end. Run the full test suite after each task to catch regressions.
