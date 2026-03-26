# LfX — Learning from Experience

Unified learning API for AI agents. Three learning layers — **harness** (prompt/playbook), **router** (model routing), **weights** (LoRA/GRPO) — all following the same protocol. Tinker/SkyRL compatible.

## Install

```bash
pip install -e .
```

For weight training (GPU):
```bash
pip install -e lfx/skyrl[fsdp]
```

## Quickstart

### Harness learning (prompt optimization, no GPU)

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Run:
```bash
python examples/train_runner.py examples/configs/math_harness.json
```

This runs the math environment with Claude Haiku as the task model and
Claude Sonnet as the reflector. The reflector analyzes failures and adds
playbook entries that improve the system prompt over iterations.

### Weight training (SkyRL GRPO, GPU)

```bash
python examples/train_runner.py examples/configs/math_weight.json
```

Uses SkyRL/Tinker to train LoRA weights via GRPO on the same math tasks.
Requires a GPU with SkyRL installed.

### Switch between modes

Same config shape. Change `"mode"` from `"harness_learning"` to `"weight"`:

```json
{"mode": "harness_learning", "env_type": "math", ...}
{"mode": "weight",           "env_type": "math", "skyrl": {...}, ...}
```

## Environments

| env_type | What it does | Needs |
|----------|-------------|-------|
| `math` | Built-in arithmetic/competition math | LLM API |
| `harbor` | [Harbor](https://harborframework.com/) sandboxed agent tasks (BFCL, etc.) | Docker + LLM API |
| `entropic` | [CRMArenaPro](https://github.com/salesforce/CRMArena) A2A benchmark | Entropic bench + LLM API |

## LLM Providers

LfX uses [litellm](https://docs.litellm.ai/) — any provider works:

```json
{"model": "anthropic/claude-haiku-4-5-20251001"}
{"model": "openai/gpt-4o-mini"}
{"model": "gemini/gemini-2.0-flash-lite"}
```

Set the provider's API key as an environment variable (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`). Or pass `api_key` and `api_base` in the
config for proxy setups.

## Examples

See [examples/README.md](examples/README.md) for all configs and Tinker
cookbook recipes.

## Architecture

```
train(config)
  -> validate_config()          # fail fast
  -> build harness + reflector  # prompt layer
  -> build weights backend      # SkyRL/Tinker (if weight mode)
  -> build env adapter          # math / harbor / entropic
  -> learning_loop()            # collect episodes, forward_backward, optim_step
```

Environments are pluggable via `ENV_BUILDERS` registry in `lfx/train.py`.

## Adding a New Environment

Write a builder function that returns `(adapter, tasks)`:

```python
# lfx/train.py
def _build_my_env(config, llm_clients):
    adapter = MyAdapter(...)  # must implement run_episode(task, agent_state) -> Episode
    tasks = ["task1", "task2"]
    return adapter, tasks

ENV_BUILDERS["my_env"] = _build_my_env
```

Your adapter's `run_episode` must return an `Episode` with messages, steps,
and an `EpisodeSummary` containing reward signals. See `lfx/envs/math.py`
(`MathAdapter`) for a minimal example (~60 lines).

## Limitations

- **`mode="full"`** (simultaneous harness + weight training) is disabled.
  The support-query data split needs rework for GRPO advantage computation.
  Use `weight` and `harness_learning` separately for now.
- **Episode construction is manual.** There is no `ProblemEnv` base class yet.
  New environments must build `Episode` objects directly. A higher-level
  abstraction (like Tinker cookbook's `ProblemEnv`) is planned.
