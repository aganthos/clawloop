#!/usr/bin/env python3
"""ClawLoop unified training runner.

Load a JSON config, call train(). One script, two modes.

    # Harness learning (prompt optimization, no GPU):
    python examples/train_runner.py examples/configs/math_harness.json

    # Weight training (SkyRL GRPO on GPU):
    python examples/train_runner.py examples/configs/math_weight.json

Tinker-compatible: weight mode uses SkyRL's training infrastructure
under the hood. ClawLoop wraps it with a unified API that lets you switch
between prompt learning and weight training by changing one field.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clawloop.train import MODE_LAYERS, TrainConfig, train


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(sys.argv[1])
    raw = json.loads(config_path.read_text())
    config = TrainConfig(**raw)

    logging.getLogger("clawloop").info(
        "mode=%s env=%s layers=%s",
        config.mode,
        config.env_type,
        MODE_LAYERS[config.mode],
    )

    agent_state, state_id = train(config)
    print(f"\nDone. Final state: {state_id.combined_hash[:12]}")


if __name__ == "__main__":
    main()
