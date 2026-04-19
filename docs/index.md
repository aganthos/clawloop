# ClawLoop

**AI agents that learn from experience.**

Your AI agents run, fail, and forget. ClawLoop closes the loop: it observes
agent-environment interactions, learns from them, and feeds improvements back
into the agent.

## Quick Start

```bash
uv sync
uv run clawloop demo math --dry-run
```

No API keys needed. The agent learns strategies, builds a playbook, and
improves across iterations.

## Three Learning Layers

| Layer | What it optimizes | How |
|-------|------------------|-----|
| **Harness** | Prompts, playbooks, tool config | Pluggable evolver backends analyze traces and improve the agent's full configuration surface |
| **Router** | Model selection | Trainable complexity scorer routes queries to the most cost-effective model |
| **Weights** | Model weights | Pluggable training backends (SkyRL/Tinker supports GRPO, PPO, SFT, DPO, LoRA, full fine-tune, and more) |

All three follow the same **Layer Protocol**: `forward_backward()` accumulates
updates without mutation, then `optim_step()` applies them atomically with
cross-layer rollback on failure.

## Integration Paths

| You have... | Start here |
|---|---|
| A Python agent | [`examples/demo_math.py`](https://github.com/aganthos/clawloop/blob/main/examples/demo_math.py) |
| An n8n or workflow platform | [`examples/n8n/`](https://github.com/aganthos/clawloop/tree/main/examples/n8n) |
| An OpenAI-compatible agent | [`examples/train_runner.py`](https://github.com/aganthos/clawloop/blob/main/examples/train_runner.py) with configs |
| Want zero-code-change learning | [`examples/openclaw_demo.py`](https://github.com/aganthos/clawloop/blob/main/examples/openclaw_demo.py) — OpenClaw transparent proxy |
| GPU resources for weight training | [`examples/recipes/`](https://github.com/aganthos/clawloop/tree/main/examples/recipes) |

## Enterprise

ClawLoop Enterprise adds premium learning backends and production
infrastructure. [Learn more](https://aganthos.com) or contact
[info@aganthos.com](mailto:info@aganthos.com).
