# LfX Recipes — Tinker-Compatible

LfX wraps [SkyRL/Tinker](https://github.com/NovaSky-AI/SkyRL) and adds a
harness layer for prompt optimization. Same recipe, two learning modes:

| `--mode` | What trains | Infrastructure |
|----------|------------|----------------|
| `weight` | LoRA weights via GRPO | SkyRL/Tinker (vLLM + FSDP2 + Ray) on GPU |
| `harness_learning` | System prompt via reflector | LLM API (no GPU needed) |

## Arithmetic RL

Teaches a model to solve `x + y = ?`. Mirrors Tinker cookbook's
[math_rl/arithmetic](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/math_rl).

```bash
# 1. Generate training data (Tinker parquet format)
python examples/recipes/arithmetic_dataset.py --output_dir ~/data/arithmetic

# 2a. Weight training (GPU) — real Tinker, model generates own rollouts
python examples/recipes/arithmetic.py --mode weight

# 2b. Harness learning (no GPU) — LfX prompt optimization
python examples/recipes/arithmetic.py --mode harness_learning
```

**Weight mode** uses SkyRL's full pipeline: vLLM generates rollouts from the
model being trained, `ArithmeticEnv` (SkyRL gym) scores them, GRPO computes
advantages, FSDP2 does the backward pass, NCCL syncs weights back to vLLM.

**Harness mode** uses LfX's learning loop: an external LLM generates responses,
the same scoring function evaluates them, the reflector LLM analyzes failures
and adds playbook entries that improve the system prompt.

## Guess the Number (multi-turn)

Multi-turn RL where the LLM guesses an integer 0-1024 via binary search.
Mirrors Tinker cookbook's
[multiplayer_rl/guess_number](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/multiplayer_rl/guess_number).

```bash
python examples/recipes/guess_number.py --mode harness_learning
```

## Files

```
recipes/
├── arithmetic.py           # Main recipe (--mode weight | harness_learning)
├── arithmetic_env.py       # SkyRL gym environment (Tinker-compatible)
├── arithmetic_dataset.py   # Data generator (Tinker parquet format)
├── guess_number.py         # Multi-turn recipe (harness_learning)
└── README.md
```
