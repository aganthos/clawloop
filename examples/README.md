# Tinker weight-tuning with ClawLoop — examples

A worked example of doing **reinforcement learning against a LoRA adapter
trained on Thinking Machines Lab's managed [Tinker](https://thinkingmachines.ai)
service**, using ClawLoop as the orchestration layer and
[OpenSpiel](https://github.com/google-deepmind/open_spiel) as a reference
environment.

> **Use Tinker with a different env.** The pieces below are env-agnostic;
> OpenSpiel is just the first adapter wired. See
> [Using Tinker with another env](#using-tinker-with-another-env) at the
> bottom for the Harbor-and-beyond story.

## Start here

The 100-line inline example, [`tinker_weight_demo.py`](tinker_weight_demo.py),
builds a `TrainConfig` from Python, runs one Blackjack training iter against
Tinker, and prints the durable Tinker checkpoint path it just wrote:

```bash
# Put your TINKER_API_KEY in clawloop/.env (gitignored; see .env.example).
uv sync --extra games
uv run python examples/tinker_weight_demo.py
```

Reads cleanly top-to-bottom. Every place you'd change to adapt the demo to
another env or another model is flagged with `[ADAPT]` comments.

## YAML recipes (longer runs)

For multi-iter runs + cross-run comparison, use YAML + `scripts/run_pilot.py`:

| File | What it does |
|---|---|
| [`blackjack_tinker_pilot.yaml`](blackjack_tinker_pilot.yaml) | Single-game Blackjack on Qwen3-8B. |
| [`blackjack_plus_2048_tinker_pilot.yaml`](blackjack_plus_2048_tinker_pilot.yaml) | **Multi-task RL in one LoRA** — Blackjack + 2048 episodes interleave into the same `forward_backward` batch; GRPO baselines stay per-scenario. |

```bash
uv run python scripts/run_pilot.py examples/blackjack_plus_2048_tinker_pilot.yaml \
    --n-iterations 3 --output-dir pilot_runs/mixed_3iter

# Live reward curves:
uv run python -m http.server 8770 --bind 127.0.0.1
# then http://localhost:8770/clawloop/static/learning_viewer.html?exp=/pilot_runs/mixed_3iter/experiment.jsonl
```

Add `--wandb-project NAME` to any run — metrics mirror through
`tinker_cookbook.utils.ml_log.setup_logging` (Neptune on the same switch
via `NEPTUNE_API_TOKEN`). No integration = Rich console tables + `experiment.jsonl` on disk.

## What each iteration does

1. Refresh `agent_state.sampling_client` from `backend.current_sampling_client()`.
2. Fan out N episode rollouts via `asyncio.gather` over `OpenSpielGameAdapter.run_episodes_batch`. Tinker's `SamplingClient.sample` returns a `ConcurrentFuture`; we wrap it with `asyncio.wrap_future` so asyncio can actually interleave them (without the wrap, `.result()` serializes everything — we measured ~7–8× speedup from the fix).
3. Each episode captures `prompt_tokens`, `sampled_tokens`, `sampling_logprobs` on `StepMeta.info` at sampling time. No tokenizer round-trip later.
4. `episodes_to_tinker_datums` applies GRPO (group-mean subtraction per `task_id = f"{game}_seed_{seed}"`), drops zero-variance groups, emits one `tinker.types.Datum` per LLM turn with prompt positions zero-padded.
5. `forward_backward` and `optim_step` submitted back-to-back as futures (both land in the same Tinker "clock cycle"), then awaited together.
6. `save_state(f"iter_{i}")` writes a durable, enumerable checkpoint (`tinker://...`); `save_weights_and_get_sampling_client` returns the fresh `SamplingClient` the next iter will use for rollouts. The durable path is listable via `RestClient.list_checkpoints` and reloadable via `load_state_with_optimizer`.

## The three files worth lifting

If you already have an RL loop and just want the Tinker bits:

- [`clawloop/weight_backends/_tinker_sdk.py`](../clawloop/weight_backends/_tinker_sdk.py) — thin typed adapter over the `tinker` SDK. One file you own; SDK drift lives in one place.
- [`clawloop/weight_backends/_tinker_exporter.py`](../clawloop/weight_backends/_tinker_exporter.py) — `list[Episode] → list[tinker.Datum]` with episode-level GRPO baselines, strict token/logprob alignment, zero-variance filtering.
- [`clawloop/weight_backends/tinker.py`](../clawloop/weight_backends/tinker.py) — `TinkerWeightsBackend` plugging into ClawLoop's `ClawLoopBackend` protocol. Dual save (ephemeral SamplingClient + durable `tinker://` path) every iter.

## Using Tinker with another env

The only contract the exporter imposes on your env is that each rollout's
`Episode` carries, for every LLM turn, a `StepMeta.info` dict with:

- `prompt_tokens: list[int]` — exact tokens the SamplingClient saw.
- `sampled_tokens: list[int]` — exact tokens it emitted.
- `sampling_logprobs: list[float]` — per-token logprobs, aligned 1:1.

`OpenSpielTaskEnvironment.run_episode` captures all three right at the
sampling call site — see `_sample_one_llm_attempt` in
`clawloop/environments/openspiel.py` for the pattern. Any env that *owns
its sampling loop* can do the same.

### Would Harbor work?

**Conceptually yes, ~50–100 LoC of wiring away.** Harbor's built-in agents
(`claude-code`, `openhands`, `aider`, Terminus 2, …) sample via LiteLLM,
which doesn't surface Tinker-compatible token IDs. Two ways to close the
gap:

1. **Ship a `TinkerAgent` in Harbor** that drives Tinker's `SamplingClient`
   directly and writes the three `StepMeta.info` fields into the trial
   trajectory. There's already a stub at
   `benchmarks/harbor/src/harbor/llms/tinker.py` that could host this.
2. **Write a `HarborTinkerAdapter` on the ClawLoop side** that wraps
   `HarborTaskEnvironment`, intercepts the agent call, and captures the
   fields during the trial run.

Either is a clean extension — the abstraction layers don't block it.
Not wired today; filed as a follow-up so whoever ships it finds the
design notes in one place.

## Scaling out

- Swap `Qwen/Qwen3-8B` for any model in Tinker's catalog via `tinker.base_model` in the YAML (or the inline config). Run `scripts/tinker_preflight.py` against your account to see the live list.
- Drop in new OpenSpiel games by appending to the `games:` list — GRPO groups stay per-scenario, reward scales don't cross-contaminate.
- Bring your own env by implementing the `AdapterLike` Protocol in `clawloop/core/loop.py` and populating the `StepMeta.info` fields above. Backend, exporter, and logger are all game-agnostic.

## Known limits (v1)

- Self-play for adversarial / hidden-info games is not wired — multi-player OpenSpiel games use a random opponent.
- OpenSpiel's canonical action strings don't always match how an LLM writes moves (chess SAN, hanabi hint codes). `_parse_move_fallback` handles common cases; per-game parsers are a clean follow-up.
- 2048's terminal reward only fires at the 2048 tile — rare at 30–50 turns. Intermediate reward shaping (score-delta, max-tile) is on the roadmap.
