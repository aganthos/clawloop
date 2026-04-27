#!/usr/bin/env python3
"""Deprecated shim — forwards to ``clawloop run <config.json>``.

The unified runner now lives in the CLI:

    uv run clawloop run examples/configs/math_harness.json
    uv run clawloop run examples/configs/math_harness.json --dry-run

Existing invocations such as ``python examples/train_runner.py <config>``
keep working: this shim prepends ``run`` to argv before dispatching.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clawloop.cli import main as cli_main


def main() -> None:
    sys.argv.insert(1, "run")
    cli_main()


if __name__ == "__main__":
    main()
