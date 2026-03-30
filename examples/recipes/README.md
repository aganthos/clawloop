# ClawLoop Recipes — Tinker-Compatible

ClawLoop wraps [SkyRL/Tinker](https://github.com/NovaSky-AI/SkyRL) and adds a
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

# 2b. Harness learning (no GPU) — ClawLoop prompt optimization
python examples/recipes/arithmetic.py --mode harness_learning
```

**Weight mode** uses SkyRL's full pipeline: vLLM generates rollouts from the
model being trained, `ArithmeticEnv` (SkyRL gym) scores them, GRPO computes
advantages, FSDP2 does the backward pass, NCCL syncs weights back to vLLM.

**Harness mode** uses ClawLoop's learning loop: an external LLM generates responses,
the same scoring function evaluates them, the reflector LLM analyzes failures
and adds playbook entries that improve the system prompt.

## Harbor BFCL (Function Calling)

Trains an agent on [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)
tasks using [Harbor](https://harborframework.com/) sandboxed execution.
See [SETUP_HARBOR.md](SETUP_HARBOR.md) for setup instructions.

```bash
# Download tasks
harbor datasets download bfcl_parity@1.0 -o ~/data/bfcl_parity

# Weight training (GPU + Docker)
python examples/recipes/harbor_bfcl.py --mode weight --task-dir ~/data/bfcl_parity

# Harness learning (no GPU, Docker + LLM API)
python examples/recipes/harbor_bfcl.py --mode harness_learning --task-dir ~/data/bfcl_parity
```

**Weight mode** uses SkyRL's Harbor integration: the model being trained serves
via vLLM's HTTP endpoint, Harbor's terminus-2 agent calls it to solve tasks in
Docker containers, the verifier scores results, GRPO trains.

**Harness mode** uses an API model (e.g. Claude Haiku) to run the Harbor trials.
The reflector analyzes failures and evolves the system prompt to improve
function calling accuracy.

## A2A CRMArena (Entropic)

Trains on [CRMArenaPro](https://github.com/salesforce/CRMArena) tasks via A2A
protocol. A purple agent (ClawLoop-controlled) interacts with a green evaluator
to solve CRM service requests. 7-dimension reward scoring.

```bash
# Harness learning (no GPU, needs LLM API + entropic bench)
python examples/recipes/a2a_crmarena.py --mode harness_learning --task-ids 0 1 2

# Weight training (GPU + LLM API for episode collection)
python examples/recipes/a2a_crmarena.py --mode weight --task-ids 0 1 2
```

Requires the entropic-crmarenapro benchmark at `benchmarks/a2a/entropic-crmarenapro/`
with its own venv (`uv sync` in that directory).

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
├── arithmetic.py           # Arithmetic RL (--mode weight | harness_learning)
├── arithmetic_env.py       # SkyRL gym environment (Tinker-compatible)
├── arithmetic_dataset.py   # Data generator (Tinker parquet format)
├── harbor_bfcl.py          # Harbor BFCL (--mode weight | harness_learning)
├── a2a_crmarena.py         # A2A CRMArena (--mode weight | harness_learning)
├── guess_number.py         # Multi-turn binary search (harness_learning)
├── README.md
└── SETUP_HARBOR.md         # Harbor setup instructions
```
