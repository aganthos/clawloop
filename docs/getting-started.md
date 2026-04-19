# Getting Started

## Installation

Requires Python 3.11+.

```bash
pip install -e .
```

For weight training (GPU):

```bash
git submodule update --init clawloop/skyrl
pip install -e clawloop/skyrl[fsdp]
```

## Try It (No API Keys)

```bash
uv run clawloop demo math --dry-run
```

This runs a complete learning loop with a mock LLM. The agent starts with
mistakes, the reflector analyzes failures, learns strategies, and injects them
into the system prompt. You'll see rewards climb toward 1.0.

## With a Real LLM

Set your API key and run:

```bash
export ANTHROPIC_API_KEY=sk-...
uv run python examples/demo_math.py
```

ClawLoop uses [litellm](https://docs.litellm.ai/) — any provider works:

```bash
export OPENAI_API_KEY=sk-...
CLAWLOOP_TASK_MODEL=openai/gpt-4o-mini \
CLAWLOOP_REFLECTOR_MODEL=openai/gpt-5.4-nano \
    uv run python examples/demo_math.py
```

## Add Learning to Your Agent

Two lines to wrap an existing LLM client:

```python
import clawloop

wrapped = clawloop.wrap(your_llm_client, collector)
result = wrapped.complete(messages)  # transparently captures traces
```

Or use the full agent API:

```python
from clawloop import ClawLoopAgent
from clawloop.environments.math import MathEnvironment

agent = ClawLoopAgent(
    task_client=task_llm,
    reflector_client=reflector_llm,
    base_system_prompt="You are a math solver.",
)
results = agent.learn(MathEnvironment(), iterations=10, episodes_per_iter=5)
```

## Config-Driven Training

No code needed — just a JSON config:

```bash
python examples/train_runner.py examples/configs/math_harness.json
```

See [`examples/configs/`](https://github.com/aganthos/clawloop/tree/main/examples/configs)
for ready-made configurations.

## LLM Providers

Any litellm-supported provider:

```json
{"model": "anthropic/claude-haiku-4-5-20251001"}
{"model": "openai/gpt-5-nano"}
{"model": "gemini/gemini-3.1-flash-lite"}
```

Set the provider's API key as an environment variable (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`), or pass `api_key` and `api_base` in
the config.

## Next Steps

- [Concepts](concepts.md) — understand the core types and architecture
- [Adding Environments](adding-environments.md) — connect your own benchmark
- [Examples README](https://github.com/aganthos/clawloop/blob/main/examples/README.md) — all integration paths
