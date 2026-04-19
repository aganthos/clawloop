"""Eval a ClawLoop Tinker adapter on a disjoint OpenSpiel seed pool.

Prints mean, stdev, sem for the returned rewards. Use before/after a training
run to verify learning (the pool is disjoint from training seeds).
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys

from clawloop.config import load_env
from clawloop.core.loop import AgentState
from clawloop.environments.openspiel import (
    OpenSpielTaskConfig, OpenSpielTaskEnvironment,
)
from clawloop.weight_backends.tinker import TinkerWeightsBackend, TinkerWeightsConfig


def _parse_seed_range(spec: str) -> list[int]:
    if ":" in spec:
        lo, hi = spec.split(":")
        return list(range(int(lo), int(hi)))
    return [int(x) for x in spec.split(",")]


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--game", required=True,
                        help="OpenSpiel game_name (blackjack, twenty_forty_eight, ...)")
    parser.add_argument("--adapter-path", default="",
                        help="tinker:// path, or empty to eval the base model.")
    parser.add_argument("--seeds", default="1000:1064",
                        help="seed range (lo:hi) or comma-separated list")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    load_env()

    backend = TinkerWeightsBackend(TinkerWeightsConfig(base_model=args.base_model))
    if args.adapter_path:
        backend.load_state({"adapter_paths": [args.adapter_path]}).result()

    cfg = OpenSpielTaskConfig(
        game_name=args.game, seeds=[0],
        max_turns=args.max_turns, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )
    seeds = _parse_seed_range(args.seeds)

    returns: list[float] = []
    illegals = 0
    for seed in seeds:
        env = OpenSpielTaskEnvironment(cfg, seed=seed)
        state = AgentState(
            sampling_client=backend.current_sampling_client(),
            renderer=backend.renderer,
            tokenizer=backend.tokenizer,
        )
        episode = await env.run_episode(state)
        returns.append(episode.summary.effective_reward())
        if episode.summary.signals.get("illegal_parse") is not None:
            sig = episode.summary.signals["illegal_parse"]
            v = sig.value if hasattr(sig, "value") else sig
            if v and v > 0:
                illegals += 1

    mean = statistics.mean(returns)
    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sem = stdev / (len(returns) ** 0.5) if returns else 0.0
    print(f"game={args.game} n={len(returns)} mean={mean:.4f} stdev={stdev:.4f} "
          f"sem={sem:.4f} illegal_rate={illegals}/{len(returns)} = "
          f"{illegals/max(len(returns),1):.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
