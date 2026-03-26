#!/usr/bin/env bash
# Setup Lambda GPU box for SkyRL integration testing.
# Usage: ssh -A ubuntu@<host> 'bash -s' < scripts/gpu_validation/setup.sh
set -euxo pipefail

# --- Python 3.12 (deadsnakes) ---
if ! command -v python3.12 &>/dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
fi
python3.12 --version

# --- uv ---
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

# --- Clone repo ---
cd ~
if [ ! -d aganthos ]; then
    git clone git@github.com:aganthos/aganthos.git
fi
cd aganthos
git fetch origin
git checkout feat/test_weight_layer
git pull origin feat/test_weight_layer

# --- Init submodules ---
git submodule update --init --recursive

# --- Create venv with Python 3.12 ---
uv venv --python python3.12 .venv
source .venv/bin/activate

# --- Install SkyRL with fsdp extra (torch, vllm, ray, flash-attn, etc.) ---
# IMPORTANT: install SkyRL first to get torch/vllm pins right
cd lfx/skyrl
uv pip install -e ".[fsdp,dev]"
cd ../..

# --- Install LfX ---
uv pip install -e ".[dev]"

# --- Pin litellm to known-good version to avoid dependency chaos ---
uv pip install "litellm==1.82.1"

# --- Prepare GSM8K data (small subset for testing) ---
python lfx/skyrl/examples/train/gsm8k/gsm8k_dataset.py \
    --output_dir ~/data/gsm8k \
    --max_train_dataset_length 100

# --- Prepare multiply data ---
python lfx/skyrl/examples/train/multiply/multiply_dataset.py \
    --output_dir ~/data/multiply \
    --train_size 200 --test_size 50

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ~/aganthos/.venv/bin/activate"
echo "Run tests with: cd ~/aganthos && bash scripts/gpu_validation/run_all.sh"
