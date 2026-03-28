# ClawLoop Examples

ClawLoop provides a **unified training API** — one `train()` call, one config, flip
`mode` to switch between prompt learning and weight training.

## Unified API (`train_runner.py` + JSON configs)

```bash
# Same runner, different configs:
python examples/train_runner.py examples/configs/math_harness.json      # prompt optimization
python examples/train_runner.py examples/configs/math_weight.json       # SkyRL GRPO on GPU
python examples/train_runner.py examples/configs/entropic_harness.json  # A2A prompt optimization
python examples/train_runner.py examples/configs/entropic_weight.json   # A2A weight training on GPU
python examples/train_runner.py examples/configs/harbor_harness.json    # Harbor prompt optimization
python examples/train_runner.py examples/configs/harbor_weight.json     # Harbor weight training on GPU
```

| `mode` | What trains | Infrastructure |
|--------|------------|----------------|
| `weight` | LoRA weights via GRPO | SkyRL/Tinker (vLLM + FSDP2 + Ray) on GPU |
| `harness_learning` | System prompt via reflector | LLM API (no GPU needed) |

### Configs

```
configs/
├── math_harness.json        # Math env, prompt optimization
├── math_weight.json         # Math env, SkyRL weight training
├── entropic_harness.json    # A2A CRMArena, prompt optimization
├── entropic_weight.json     # A2A CRMArena, weight training
├── harbor_harness.json      # Harbor BFCL, prompt optimization
└── harbor_weight.json       # Harbor BFCL, weight training
```

All configs follow the same schema (`TrainConfig`). The only difference between
harness and weight variants is `mode` and the presence of `skyrl` config.

## OpenClaw Proxy — Improve any agent with zero code changes

ClawLoop can improve any OpenAI-compatible agent (pi-mono, LangChain, CrewAI, raw
API calls) by sitting as a transparent proxy between the agent and the LLM:

```
Agent ──► ClawLoop Proxy ──► Upstream LLM
            │
            ├─ inject playbook skills
            ├─ forward request
            ├─ stream response back
            ├─ capture trace for training
            └─ strip skills before storage
```

The agent requires **zero code changes** — just point `base_url` at the proxy.

Requires Node.js and an OpenAI-compatible API (OpenAI, Anthropic via proxy,
vLLM, etc.). Note: Gemini's OpenAI-compatible endpoint has SSE format
differences that cause empty responses with pi-mono.

```bash
# Install pi-mono runner (one time)
cd scripts/openclaw_runner && npm install && cd ../..

# Run the demo
UPSTREAM_URL=https://api.openai.com/v1 UPSTREAM_KEY=$OPENAI_API_KEY \
    PYTHONPATH=. python examples/openclaw_proxy_demo.py
```

See [`openclaw_proxy_demo.py`](openclaw_proxy_demo.py) for the full annotated example.

## Tinker Cookbook Recipes

Self-contained scripts that mirror
[Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
patterns. These use the ClawLoop layer API directly for more control.

See [recipes/README.md](recipes/README.md) for details.

## Tested End-to-End

| Env | harness_learning | weight |
|-----|:---:|:---:|
| Math | Local Mac (Gemini) | Lambda A10 (Gemini + SkyRL) |
| Harbor BFCL | Lambda (Gemini + Docker) | Lambda (Oracle + Docker + SkyRL) |
| Entropic A2A | Local Mac (Gemini) | Lambda A10 (Gemini + SkyRL) |
| OpenClaw Proxy | Local Mac (OpenAI / Anthropic) | — |
