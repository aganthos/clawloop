"""Full E2E learning loop test via CLIProxyAPI.

PRIVATE — not synced to public clawloop repo (enterprise_clawloop/ not in .publicpaths).

Same demo as examples/openclaw_demo.py but pre-configured for
CLIProxyAPI (free, no API key needed) with Haiku 4.5.

Usage:
    PYTHONPATH=. python enterprise_clawloop/examples/openclaw_cliproxy.py
"""
import os
import sys

# Auto-detect CLIProxyAPI
try:
    import httpx
    r = httpx.get(
        "http://127.0.0.1:8317/v1/models",
        headers={"Authorization": "Bearer kuhhandel-bench-key"},
        timeout=2,
    )
    if r.status_code != 200:
        print("CLIProxyAPI not responding on :8317")
        sys.exit(1)
except Exception:
    print("CLIProxyAPI not running. Start it first.")
    sys.exit(1)

os.environ.setdefault("UPSTREAM_URL", "http://127.0.0.1:8317/v1")
os.environ.setdefault("UPSTREAM_KEY", "kuhhandel-bench-key")
os.environ.setdefault("MODEL", "claude-haiku-4-5-20251001")

# Delegate to the public demo
import importlib.util
spec = importlib.util.spec_from_file_location(
    "openclaw_demo",
    os.path.join(os.path.dirname(__file__), "..", "examples", "openclaw_demo.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()
