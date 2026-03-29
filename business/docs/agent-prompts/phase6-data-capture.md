# Agent Prompt: Phase 6 — Data Capture / Evolution Archive

Implement Phase 6: Evolution Archive — structured data capture for all ClawLoop learning runs. This data seeds future learned evolvers and cross-domain transfer.

Working directory: /Users/robertmueller/Desktop/aganthos
Branch: create `feat/evolution-archive` from main (pull latest first)

## Context

ClawLoop already has:
- clawloop/core/evolver.py — Evolver interface with EvolverResult, HarnessSnapshot, Provenance
- clawloop/evolvers/local.py — LocalEvolver wrapping Reflector + GEPA + Paradigm
- clawloop/core/loop.py — the learning loop with ExperimentLog (JSONL per iteration)

We need a structured archive that captures (agent_state, traces, improvement_actions, reward_delta) tuples. Start simple: community gets local storage (SQLite + JSONL).

## What to build

### 1. Schema types (clawloop/archive/schema.py)

Four core entities as dataclasses with .to_dict() and .from_dict() classmethod:

RunRecord:
  run_id (str, UUID), bench (str), domain_tags (list[str]),
  agent_config (dict — system prompt + playbook + model + tools),
  n_iterations (int), best_reward (float), improvement_delta (float),
  total_cost_tokens (int), parent_run_id (str | None, for lineage),
  config_hash (str, SHA256 of canonical agent config),
  created_at (float), completed_at (float | None)

IterationRecord:
  run_id (str), iteration_num (int),
  harness_snapshot_hash (str, SHA256 of HarnessSnapshot.to_dict()),
  mean_reward (float), reward_trajectory (list[float]),
  evolver_action (dict — serialized EvolverResult),
  cost_tokens (int),
  created_at (float)

EpisodeRecord:
  run_id (str), iteration_num (int), episode_id (str), task_id (str),
  bench (str), model (str),
  reward (float), signals (dict), n_steps (int), n_tool_calls (int),
  token_usage (dict), latency_ms (int),
  messages_ref (str — path to full trace file)

AgentVariant:
  variant_hash (str, SHA256), system_prompt (str),
  playbook_snapshot (dict), model (str), tools (list),
  first_seen_run_id (str)

### 2. Content hashing (clawloop/archive/content_hash.py)

canonical_hash(obj: dict) -> str:
  - Sort keys recursively
  - json.dumps with sort_keys=True, separators=(',', ':')
  - SHA256 hex digest
  - Must be deterministic (same input → same hash always)

### 3. Archive store (clawloop/archive/store.py)

Protocol: ArchiveStore with methods:
  log_run_start(run: RunRecord)
  log_iteration(iteration: IterationRecord)
  log_episodes(episodes: list[EpisodeRecord])
  log_run_complete(run_id: str, best_reward: float, improvement_delta: float)
  get_run(run_id: str) -> RunRecord | None
  get_similar_runs(config_hash: str, domain_tags: list[str], limit: int) -> list[RunRecord]

Two implementations:

LocalArchiveStore:
  - SQLite (WAL mode) for metadata (runs, iterations, variants)
  - JSONL files for episode records (one file per run: {run_id}/episodes.jsonl)
  - Full traces to {output_dir}/traces/{episode_id}.json
  - Default location: ~/.clawloop/archive/
  - Tables: runs, iterations, episodes, variants
  - Auto-create schema on first use

NullArchiveStore:
  - No-op implementation, for when archiving is disabled
  - All methods return None / do nothing

### 4. Wire into learning loop (modify clawloop/core/loop.py)

Add optional `archive: ArchiveStore | None = None` parameter to learning_loop().
- On loop start: log_run_start() with initial agent config
- After episodes collected per iteration: log_episodes() (metadata only, full traces to disk)
- After each optim_step: log_iteration() with HarnessSnapshot hash + EvolverResult
- On loop end: log_run_complete() with best_reward and improvement_delta

Also wire into AsyncLearner (clawloop/learner.py) — same pattern via optional archive parameter.

### 5. Export utility (clawloop/archive/export.py)

export_to_parquet(archive_dir: str, output_path: str):
  - Read SQLite + JSONL
  - Export runs, iterations, episodes as Parquet files
  - For ML training pipelines
  - Requires pyarrow (optional dependency)

### 6. Package init (clawloop/archive/__init__.py)

Export: ArchiveStore, LocalArchiveStore, NullArchiveStore, RunRecord, IterationRecord, EpisodeRecord, AgentVariant, canonical_hash

### Tests

tests/test_archive_schema.py — serialization roundtrips, hash determinism
tests/test_archive_store.py — LocalArchiveStore CRUD with temp SQLite
tests/test_archive_export.py — Parquet export from test data (skip if pyarrow not installed)
tests/test_archive_integration.py — learning_loop with archive=LocalArchiveStore, verify data captured

### Dependencies

Add to pyproject.toml optional dependencies:
  archive = ["pyarrow>=14.0"]  # only needed for Parquet export

SQLite is stdlib — no dependency needed for core archive.

## Commit sequence

1. `feat: archive schema types with content hashing`
2. `feat: LocalArchiveStore with SQLite + JSONL storage`
3. `feat: wire archive into learning loop and AsyncLearner`
4. `feat: Parquet export for ML training`
5. `feat: export archive types from clawloop`

## Rules
- Commit format: `fix:`, `feat:`, or `chore:` + one line. NO Co-Authored-By. NO multi-line bodies.
- Run pytest tests/ -x -q after each chunk
- Run bash scripts/audit_public.sh before final commit
- This is PUBLIC code (inside clawloop/) — no enterprise secrets
- SQLite schema should use WAL mode for concurrent reads
- All timestamps as float (time.time()) for consistency with existing Episode
