# ClawLoop Examples

## Start Here

**Zero setup, 10 seconds:**
```bash
python examples/demo_math.py --dry-run
```

**See the playbook learning internals:**
```bash
python examples/playbook_demo.py --dry-run
```

Both run with mock LLMs — no API keys, no network, finishes instantly.

## Choose Your Path

### I have a Python agent

Use `ClawLoopAgent` with any litellm-supported LLM:

```bash
# With Anthropic
ANTHROPIC_API_KEY=... python examples/demo_math.py

# With OpenAI
CLAWLOOP_TASK_MODEL=openai/gpt-4o-mini CLAWLOOP_REFLECTOR_MODEL=openai/gpt-4o \
    python examples/demo_math.py
```

For deeper control, see [`playbook_demo.py`](playbook_demo.py) — walks through
the two-phase protocol (forward_backward → optim_step), playbook scoring,
structured skill entries, and tag-based retrieval step by step.

### I use n8n or a workflow platform

ClawLoop integrates with n8n via webhooks — no Python code in your workflow.
An n8n workflow sends tickets to the LLM, posts traces to clawloop-server,
and the server learns in the background.

See [`n8n/README.md`](n8n/README.md) for setup, the importable workflow, and
a demo script that shows a customer support agent improving across rounds.

### I have an OpenAI-compatible agent or API

Use the unified `train_runner.py` with JSON configs. Same runner, different
environments:

```bash
python examples/train_runner.py examples/configs/math_harness.json        # math, prompt optimization
python examples/train_runner.py examples/configs/entropic_harness.json    # CRMArena A2A, prompt optimization
python examples/train_runner.py examples/configs/harbor_harness.json      # Harbor BFCL, prompt optimization
```

Switch to weight training by changing `mode`:

```bash
python examples/train_runner.py examples/configs/math_weight.json         # math, SkyRL on GPU
python examples/train_runner.py examples/configs/entropic_weight.json     # CRMArena, weight training
python examples/train_runner.py examples/configs/harbor_weight.json       # Harbor, weight training
```

All configs follow the same `TrainConfig` schema. See [`configs/`](configs/)
for the full set.

### I want zero-code-change learning (OpenClaw)

The OpenClaw proxy sits transparently between any OpenAI-compatible agent
and its LLM. It captures traces, learns from them, and injects playbook
skills — without touching agent code:

```bash
cd examples/openclaw_runner && npm install && cd ../..

UPSTREAM_URL=https://api.openai.com/v1 UPSTREAM_KEY=$OPENAI_API_KEY \
    PYTHONPATH=. python examples/openclaw_demo.py
```

Requires Node.js and an OpenAI-compatible Chat Completions endpoint.
See [`openclaw_demo.py`](openclaw_demo.py) for the full annotated example.

### I have GPU resources for weight training

The [recipes/](recipes/) directory contains SkyRL/Tinker-compatible scripts
for GRPO, PPO, and full fine-tuning:

- **Arithmetic RL** — `recipes/arithmetic.py` (mirrors Tinker cookbook)
- **Harbor BFCL** — `recipes/harbor_bfcl.py` (function calling in Docker)
- **A2A CRMArena** — `recipes/a2a_crmarena.py` (multi-agent CRM tasks)
- **Guess the Number** — `recipes/guess_number.py` (multi-turn RL)

See [`recipes/README.md`](recipes/README.md) for setup and details.

## Mode Reference

| `mode` | What trains | Infrastructure |
|--------|------------|----------------|
| `harness_learning` | System prompt via reflector | LLM API (no GPU) |
| `weight` | Model weights via RL / fine-tuning | SkyRL/Tinker (vLLM + FSDP2 + Ray) on GPU |

## Tested End-to-End

| Env | harness_learning | weight |
|-----|:---:|:---:|
| Math | Gemini | Gemini + SkyRL |
| Harbor BFCL | Gemini + Docker | Oracle + Docker + SkyRL |
| Entropic A2A | Gemini | Gemini + SkyRL |
| OpenClaw Proxy | OpenAI | -- |
| n8n Integration | any provider | -- |
