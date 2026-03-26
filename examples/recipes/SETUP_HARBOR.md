# Harbor Setup for ClawLoop Recipes

## Prerequisites

- Python >= 3.12
- Docker running (`docker ps` should work)
- For weight mode: GPU + SkyRL installed (`pip install -e clawloop/skyrl[fsdp]`)
- For harness mode: LLM API access (CLIProxyAPI or direct key)

## 1. Install Harbor

```bash
pip install harbor
# or
uv pip install harbor
```

Verify:
```bash
harbor --version
python -c "from harbor.trial.trial import Trial; print('Harbor OK')"
```

## 2. Download BFCL Tasks

**Option A: Registered dataset (recommended)**
```bash
# Small parity subset (123 tasks, fast)
harbor datasets download bfcl_parity@1.0 -o ~/data/bfcl_parity

# Full BFCL (3641 tasks)
harbor datasets download bfcl@1.0 -o ~/data/bfcl
```

**Option B: Use existing fixtures (3 tasks, for smoke testing)**
```bash
# Already in the repo
ls tests/fixtures/harbor_tasks/
# bfcl-fail-0  bfcl-simple-0  bfcl-simple-1
```

## 3. Run BFCL Recipe

### Harness learning (no GPU)

Optimizes the system prompt via reflector LLM. Harbor runs real agent trials
in Docker containers.

```bash
# Using fixtures (smoke test)
python examples/recipes/harbor_bfcl.py \
    --mode harness_learning \
    --task-dir tests/fixtures/harbor_tasks \
    --iterations 2 --episodes 2

# Using full dataset
python examples/recipes/harbor_bfcl.py \
    --mode harness_learning \
    --task-dir ~/data/bfcl_parity \
    --iterations 5 --episodes 10
```

Requires: Docker + LLM API (CLIProxyAPI at localhost:8317 or direct key).

### Weight training (GPU)

Real Tinker/SkyRL GRPO training. The model being trained generates responses
via vLLM, Harbor runs them in Docker, GRPO trains on the rewards.

```bash
python examples/recipes/harbor_bfcl.py \
    --mode weight \
    --task-dir ~/data/bfcl_parity \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --iterations 1
```

Requires: GPU + Docker + SkyRL[fsdp] installed.

## 4. Environment Providers

By default Harbor uses Docker (local). For cloud sandboxes:

| Provider | Setup |
|----------|-------|
| Docker | `docker ps` must work. No extra config. |
| Daytona | `export DAYTONA_API_KEY=...` |
| Modal | `export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...` |

To change provider, edit the `trial_config` in the recipe or pass
`--env-type daytona`.

## 5. Lambda GPU Box Setup

```bash
# Already done if you followed scripts/gpu_validation/setup.sh
# Just add Harbor:
source ~/aganthos/.venv/bin/activate
uv pip install harbor

# Docker should already be running
docker ps

# Download tasks
harbor datasets download bfcl_parity@1.0 -o ~/data/bfcl_parity

# Run weight training
cd ~/aganthos
PYTHONPATH=$PWD:$PYTHONPATH python examples/recipes/harbor_bfcl.py \
    --mode weight \
    --task-dir ~/data/bfcl_parity \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

## Troubleshooting

**`ImportError: harbor` not found**
→ `pip install harbor` (requires Python >= 3.12)

**Docker permission denied**
→ `sudo usermod -aG docker $USER && newgrp docker`

**Harbor trial timeout**
→ Increase timeout: edit `trial_config["agent"]["override_timeout_sec"]` in recipe

**BFCL download fails**
→ Check `harbor datasets list` for available versions. Use fixtures as fallback.

**vLLM OOM in weight mode**
→ Reduce `gpu_memory_utilization` (default 0.5) or use a smaller model.
