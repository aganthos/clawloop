"""Run a ClawLoop training pilot from a YAML config.

Generic — pass any config; --n-iterations overrides the YAML for smoke tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from clawloop.config import load_env
from clawloop.train import TrainConfig, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="override n_iterations (e.g. 1 for smoke test)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="override output_dir for logs")
    parser.add_argument("--wandb-project", default=None,
                        help="if set, mirror metrics to this wandb project (requires WANDB_API_KEY)")
    parser.add_argument("--wandb-name", default=None,
                        help="optional wandb run name (defaults to output_dir basename)")
    args = parser.parse_args()

    load_env()  # pick up TINKER_API_KEY (and WANDB_API_KEY if present) from clawloop/.env

    raw = yaml.safe_load(args.config.read_text())
    if args.n_iterations is not None:
        raw["n_iterations"] = args.n_iterations
    if args.output_dir is not None:
        raw["output_dir"] = str(args.output_dir)
    if args.wandb_project is not None:
        raw["wandb_project"] = args.wandb_project
    if args.wandb_name is not None:
        raw["wandb_name"] = args.wandb_name
    cfg = TrainConfig.model_validate(raw)
    result = train(cfg)
    print("done", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
