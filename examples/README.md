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

## Tinker Cookbook Recipes

Self-contained scripts that mirror
[Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
patterns. These use the ClawLoop layer API directly for more control.

See [recipes/README.md](recipes/README.md) for details.

## Tested End-to-End

| Env | harness_learning | weight |
|-----|:---:|:---:|
| Math | Local Mac (CLIProxyAPI) | Lambda A10 (Gemini + SkyRL) |
| Harbor BFCL | Lambda (Gemini + Docker) | Lambda (Oracle + Docker + SkyRL) |
| Entropic A2A | Local Mac (CLIProxyAPI) | Lambda A10 (Gemini + SkyRL) |
