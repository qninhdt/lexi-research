#!/usr/bin/env bash
set -euo pipefail

echo "=== [tau-research] Step 1: Checking GPU hardware ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No NVIDIA GPU detected. Running in CPU mode."
fi

echo "=== [tau-research] Step 2: Installing uv and project dependencies ==="
pip install -q uv
uv sync --extra colab

echo "=== [tau-research] Step 3: Cloning tau2-bench (pinned v1.0.1) ==="
mkdir -p third_party
if [ ! -d "third_party/tau2-bench" ]; then
    git clone --branch v1.0.1 https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench
fi

echo "=== [tau-research] Step 4: Verifying environment ==="
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
uv run python -c "import tau_research; print(f'tau-research: {tau_research.__version__}')"

echo "=== [tau-research] Setup complete! ==="
