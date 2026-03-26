#!/usr/bin/env bash
# Run all SkyRL integration validation tests on a GPU box.
# Usage: cd ~/aganthos && bash scripts/gpu_validation/run_all.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd ~/aganthos
source .venv/bin/activate

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  $name"
    echo "================================================================"
    if "$@"; then
        echo "  >>> PASSED: $name"
        PASS=$((PASS + 1))
    else
        echo "  >>> FAILED: $name (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

# 0. Unit tests (including previously-skipped SkyRL compat tests)
run_test "pytest: SkyRL compat + export + backend tests" \
    python -m pytest tests/test_skyrl_compat.py tests/test_skyrl_export.py tests/test_skyrl_backend.py -v

# 1. LfX integration: export pipeline (no GPU needed, fast)
run_test "LfX export pipeline -> PreparedModelPassBatch" \
    python scripts/gpu_validation/test_lfx_integration.py

# 2. SkyRL native: GSM8K GRPO LoRA 0.5B (single A10)
run_test "SkyRL native: GSM8K 0.5B LoRA GRPO" \
    bash scripts/gpu_validation/test_skyrl_native_math.sh

# 3. SkyRL native: Multiply env with multi-turn RL (single A10)
run_test "SkyRL native: Multiply env 0.5B" \
    bash scripts/gpu_validation/test_skyrl_multiply.sh

echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Skipped: $SKIP"
echo "================================================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
