# Changelog

## Unreleased

- Evolver protocol: unified Reflector + GEPA + Paradigm behind `Evolver` interface
- OpenClaw adapter: transparent proxy for any OpenAI-compatible agent
- n8n integration: webhook-based learning for workflow platforms
- Evolution archive: LocalArchiveStore (SQLite WAL + JSONL) with Parquet export
- Playbook curator: lightweight mode for narrow agents, batch promotion
- Reward composition: priority-based signal merging (user > outcome > execution > judge)
- Improved onboarding: `--dry-run` demos, architecture diagram, integration path routing

## 0.0.1 (2026-03-27)

Initial public release.

- Three learning layers: Harness (prompt/playbook), Router (model selection), Weights (RL/finetune)
- Unified Layer Protocol with atomic rollback and regression gating
- Live mode: `clawloop.wrap()` + `EpisodeCollector` + `AsyncLearner`
- Benchmark adapters: CRMArena, Harbor
- SkyRL backend for weight training
- OTel/OpenInference export
- HTTP integration server
