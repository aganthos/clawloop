# Agent Prompt: WP2 — AgentBeats Competition

## Your Task

Prepare and submit lfx-powered purple agents to the AgentBeats competition. The submission
artifact is a **frozen harness** (learned system prompt + learned playbook) baked into an
A2A-compatible purple agent.

## Context

**lfx** learns from agent episodes. The Harness layer evolves system prompts (GEPA Pareto front)
and accumulates playbook entries (strategies learned from failures). After training, we freeze
the best prompt + playbook and deploy it as the competition agent.

**AgentBeats** is a Berkeley-run competition with sprint-based tracks. Purple agents are
registered on agentbeats.dev and evaluated on held-out tasks via A2A protocol.

## What Already Exists

- `adapters/car.py` + `_car_purple.py` + `_car_rewards.py` — CAR-bench adapter (merged, working)
- `adapters/entropic.py` + `_entropic_purple.py` + `_entropic_rewards.py` — CRM Arena adapter
  (on branch `feat/entropic-crmarena-bench`, fully implemented, +1323 lines, 3 test files)
- `adapters/tau2.py` — stub only (needs implementation for Sprint 2)
- `configs/entropic_train.json` + `entropic_smoke.json` — training configs
- CLI: `lfx run --bench entropic`, `lfx eval --bench entropic`, `lfx setup-bench --bench entropic`

## Step 1: Merge and Smoke Test

```bash
git merge feat/entropic-crmarena-bench
git submodule update --init benchmarks/a2a/entropic-crmarenapro
lfx setup-bench --bench entropic
lfx run --bench entropic --config configs/entropic_smoke.json
```

Verify: smoke run completes, episodes are collected, experiment.jsonl is written.

## Step 2: Training Run

```bash
lfx run --bench entropic --config configs/entropic_train.json --output runs/crm-v1/
```

Training config should specify: iterations (start with 10 for validation, then 50-100 for full),
model for purple agent, model for reflector, seed for reproducibility.

Monitor `runs/crm-v1/experiment.jsonl` during training:
- Is average reward trending up?
- Is playbook growing? Are entries being pruned/merged by curator?
- Are GEPA candidates being generated and selected?

## Step 3: Export Frozen Artifacts

After training, the output directory should contain:
- `experiment.jsonl` — per-iteration metrics
- `playbook.json` — `harness.playbook.to_dict()`
- `best_prompt.txt` — `harness.system_prompts[bench]`
- `config.json` — full training config used
- `git_hash.txt` — `git rev-parse HEAD`
- `manifest.json` — SHA256 hashes of all above files

If the CLI doesn't produce all of these automatically, create a script `scripts/export_submission.py`
that reads the training output and produces the complete submission package.

## Step 4: Baseline Comparison

Run the same benchmark with a base agent (no lfx harness) and compare:

```bash
# Baseline (no playbook, no GEPA prompt)
lfx eval --bench entropic --config configs/entropic_smoke.json --output runs/crm-baseline/

# Compare
python scripts/compare_runs.py runs/crm-baseline/ runs/crm-v1/
```

**Gate**: harness-tuned must beat baseline by >5% on the benchmark's primary composite score.
Statistical fallback: submit if non-worse at 95% CI (for noisy benchmarks).

If the comparison script doesn't exist, create it — read experiment.jsonl from both runs,
compute per-task scores, run a paired t-test or bootstrap CI.

## Step 5: Register on AgentBeats

The purple agent needs to be A2A-compatible and registered on agentbeats.dev. The existing
`_entropic_purple.py` is already an A2A Starlette server with `message/send` JSON-RPC
and `/.well-known/agent.json`. Follow the same pattern as `_car_purple.py`.

## Step 6: τ²-bench Adapter (Sprint 2)

Same pattern as CAR/Entropic:
- `adapters/tau2.py` — implement the stub (currently raises NotImplementedError)
- `adapters/_tau2_purple.py` — A2A purple agent with harness injection
- `adapters/_tau2_rewards.py` — map τ²-bench metrics to lfx RewardSignal
- `configs/tau2_train.json` + `tau2_smoke.json`
- Tests: `test_tau2_adapter.py`, `test_tau2_purple.py`, `test_tau2_rewards.py`

τ²-bench evaluates airline/retail customer service agents. Sprint 2 deadline: Apr 12.

## Commit Style

- Format: `fix:`, `feat:`, or `chore:` + one line description
- NO Co-Authored-By, NO multi-line bodies

## Testing

```bash
cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -x -v
```
