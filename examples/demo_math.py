#!/usr/bin/env python3
"""Thin shim for clone-based usage.

The real implementation lives in clawloop/demo_math.py so it is available
after a wheel install.  This file exists only so that repo users can run:

    python examples/demo_math.py --dry-run

Installed users should use:
    clawloop demo math --dry-run
    python -m clawloop.demo_math --dry-run
"""

from __future__ import annotations

import os
import sys

# Allow running from the repo root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clawloop.demo_math import main  # noqa: E402

if __name__ == "__main__":
    main()
