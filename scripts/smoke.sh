#!/usr/bin/env bash
# smoke.sh — Quick pre-release sanity check.
# Runs the no-key demo, tests, package build, wheel install, and audit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "===  1/5  No-key math demo (dry-run)  ==="
uv run clawloop demo math --dry-run

echo ""
echo "===  2/5  Unit tests  ==="
uv run pytest tests/ -x -q --timeout=60

echo ""
echo "===  3/5  Package build  ==="
rm -rf dist/
uv sync --extra release --quiet
uv run python -m build --outdir dist/

echo ""
echo "===  4/5  Wheel install check  ==="
WHEELS=(dist/*.whl)
uv pip install "${WHEELS[0]}" --force-reinstall --quiet
uv run python -c "import clawloop; print(f'clawloop {clawloop.__version__} imported OK')"

echo ""
echo "===  5/5  Package audit (twine check)  ==="
uv run twine check dist/*

echo ""
echo "✅  All smoke checks passed."
