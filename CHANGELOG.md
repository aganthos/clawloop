# Changelog

## 0.0.1 (2026-03-27)

Initial public release.

- Three learning layers: Harness (prompt optimization), Router (model selection), Weights (training)
- Unified Layer Protocol with atomic rollback and regression gating
- Live mode: `clawloop.wrap()` + `EpisodeCollector` + `AsyncLearner`
- Benchmark adapters: CRM Arena, tau2-bench (stub), Harbor
- SkyRL backend for LoRA/GRPO training
- OTel/OpenInference export
- HTTP integration server
