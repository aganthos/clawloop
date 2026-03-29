# Agent Prompt: PR2 — GEPA Completion + Learning Quality

## Your Task

Implement PR2 ("GEPA Completion + Learning Quality") from `docs/plans/2026-03-17-competitive-learnings-roadmap.md`. That file is your implementation spec — read it first, focus on the PR2 section (Tasks 2.1–2.6).

## Branch

Create and work on `feat/gepa-learning-quality` off `main`. **PR1 (`feat/playbook-curator`) must be merged first** — this PR depends on the curator, attribution, decay, and generation versioning introduced there.

## Project Context

**lfx** = "Learning from Experience" — a unified learning API for AI agents. Three layers:
- **Harness** (`layers/harness.py`) — prompt optimization via GEPA Pareto-front + ACE-style playbook memory
- **Router** (`layers/router.py`) — model selection
- **Weights** (`layers/weights.py`) — LoRA/GRPO training via SkyRL/Harbor

After PR1, the playbook has conflict/redundancy resolution, per-entry attribution, temporal decay, generation versioning, and a `PlaybookCurator`. What's still missing: GEPA has no mutation/crossover operators (it's a selection-only archive), the learning loop feeds all episodes to all layers indiscriminately (no support-query separation), there's no activity-aware scheduling, and no unified background job system.

## Key Files You'll Modify

| File | What it does | What you'll change |
|------|-------------|-------------------|
| `core/loop.py` | Learning loop orchestrator | Support-query separation: failures→harness, successes→weights, both→router. Generation flush. Fix `tried_paradigms=[]` TODO (line 308). |
| `learner.py` | `AsyncLearner` — background learning thread | Same support-query separation for live mode. Generation flush on buffer. |
| `core/intensity.py` | `AdaptiveIntensity` — controls when reflector fires | Add `cooldown_after_request`, `record_user_activity()`. Defer learning during active use. |
| `wrapper.py` | `lfx.wrap()` SDK wrapper | Wire `record_user_activity()` on incoming user messages. |
| `collector.py` | `EpisodeCollector` for live mode | Wire `record_user_activity()` on ingest. |

## New Files You'll Create

| File | Purpose |
|------|---------|
| `core/evolution.py` | `PromptEvolver` with `mutate()` and `crossover()` — LLM-based GEPA operators |
| `core/background.py` | `BackgroundScheduler`, `BackgroundTask` protocol, `BackgroundState`, `EpisodeDreamer` |
| `tests/test_support_query.py` | Failures go to harness only, successes to weights only |
| `tests/test_generation_flush.py` | Generation advance flushes stale episodes from weights buffer |
| `tests/test_evolution.py` | Mutation produces valid candidates with lineage, crossover combines |
| `tests/test_activity_intensity.py` | Cooldown defers reflection, expired cooldown allows it |
| `tests/test_background.py` | Scheduler runs tasks when conditions met, skips when not |

## Architecture Principles

- **Support-query separation** (from MetaClaw paper): failures (`effective_reward() < 0`) ONLY drive playbook evolution via Reflector. Successes (`>= 0`) ONLY drive Weights/RL training. Router gets both. This prevents RL from learning pre-adaptation behaviors that the playbook has already corrected.
- **Generation flush**: when `playbook_generation` advances (structural change from insights), flush all episodes in the Weights training buffer stamped with older generation. No sliding window — generation boundary is the natural cutoff.
- **Activity-aware intensity**: don't run `optim_step` / reflector during active user interaction. Simple cooldown timer (default 30s after last user request). Not as elaborate as MetaClaw's OMLS (no calendar integration) — just `time.time()` check.
- **Unified background scheduler**: one `BackgroundScheduler` with pluggable `BackgroundTask` protocol. Both `PlaybookConsolidation` (from PR1's curator) and `EpisodeDreamer` (cross-episode pattern analysis) are tasks. Add `in_progress` guard to prevent overlapping runs.
- **GEPA mutation**: LLM reads failing episodes + current prompt, proposes targeted fix. New candidate gets `parent_id` and `generation+1`. Crossover: LLM combines strengths of two non-dominated candidates. Both produce `PromptCandidate` objects that feed into the existing `ParetoFront`.

## Key Data Flow Change

**Before (current):**
```
episodes → ALL layers get ALL episodes
```

**After:**
```
episodes → split by effective_reward()
  reward < 0  → harness.forward_backward(support_datum)   # learn from failures
  reward >= 0 → weights.forward_backward(query_datum)      # optimize from successes
  all         → router.forward_backward(full_datum)         # needs both signals
```

This change affects both `learning_loop()` in `core/loop.py` and `AsyncLearner._learn()` in `learner.py`.

## Existing Code to Understand

- `PromptCandidate` and `ParetoFront` in `layers/harness.py` — the GEPA archive you're extending with mutation/crossover
- `AdaptiveIntensity` in `core/intensity.py` — the scheduler you're extending with activity awareness
- `PlaybookCurator.consolidate()` in `core/curator.py` (from PR1) — becomes a `BackgroundTask`
- `Harness.playbook_generation` (from PR1) — the generation counter you'll use for flush logic

## Testing

```bash
cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -x -v
```

All existing tests MUST continue to pass. Use `MockLLMClient` (from `llm.py`) for evolution tests. Use `MockEmbedding` (from `core/embeddings.py`, added in PR1) where needed.

## Commit Style

- Format: `fix:`, `feat:`, or `chore:` + one line description
- NO Co-Authored-By lines in commits
- NO multi-line bodies
- Small, focused commits — one per task or logical unit

## Execution Order

Follow the task numbers (2.1 → 2.6). Task 2.5 (agentic reflector) is a stretch goal — skip if the other tasks take long. Write tests alongside implementation. Run full suite after each task.
