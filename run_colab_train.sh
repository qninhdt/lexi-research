#!/usr/bin/env bash
# Smoke-test the QLoRA training pipeline on a free Colab GPU.
#
# Usage:  bash run_colab_train.sh [GPU]
#         bash run_colab_train.sh T4       # default
#         bash run_colab_train.sh A100
#
# Requires: google-colab-cli (`uv tool install google-colab-cli`)

set -euo pipefail

GPU="${1:-T4}"
SESSION="lexi-train"

echo "═══════════════════════════════════════════════════════"
echo "🚀  Lexi-Research — Colab Training Smoke Test"
echo "    GPU: $GPU   Session: $SESSION"
echo "═══════════════════════════════════════════════════════"

# ── 1. Provision ──────────────────────────────────────────
echo ""
echo "--- [1/5] Provisioning Colab VM (GPU=$GPU) ---"
colab new -s "$SESSION" --gpu "$GPU"

# ── 2. GPU check ──────────────────────────────────────────
echo ""
echo "--- [2/5] Verifying GPU ---"
echo "import subprocess; print(subprocess.check_output(['nvidia-smi']).decode())" \
  | colab exec -s "$SESSION"

# ── 3. Clone repo & install deps ─────────────────────────
echo ""
echo "--- [3/5] Cloning repo & installing dependencies ---"
cat <<'PYSETUP' | colab exec -s "$SESSION"
import subprocess, sys

cmds = [
    "git clone https://github.com/qninhdt/lexi-research.git /content/lexi-research",
    "pip install -q -r /content/lexi-research/requirements-colab.txt",
    "pip install -q pyarrow pydantic jinja2 openai httpx fastapi uvicorn",
]
for cmd in cmds:
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)
print("\n✅ Environment ready")
PYSETUP

# ── 4. Train ─────────────────────────────────────────────
echo ""
echo "--- [4/5] Running QLoRA training (1 epoch, 32 samples) ---"
cat <<'PYTRAIN' | colab exec -s "$SESSION"
import subprocess, sys, os
os.chdir("/content/lexi-research")
os.environ["WANDB_DISABLED"] = "true"

cmd = [
    sys.executable, "-m", "lexi_research.cli", "train", "sft",
    "--train", "ops/fixtures/smoke_50.jsonl",
    "--output", "models/smoke_test",
    "--override", "train.epochs=1",
]
print(f">>> {' '.join(cmd)}")
result = subprocess.run(cmd)
if result.returncode != 0:
    print("❌ Training failed!")
    sys.exit(result.returncode)
print("\n✅ Training complete!")
PYTRAIN

# ── 5. Download & cleanup ────────────────────────────────
echo ""
echo "--- [5/5] Downloading results & stopping VM ---"
mkdir -p models/smoke_test
colab download -s "$SESSION" /content/lexi-research/models/smoke_test models/smoke_test \
  || echo "⚠️  Download failed or no output files"
colab stop -s "$SESSION"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🎉  Done! Check models/smoke_test/ for adapter weights."
echo "═══════════════════════════════════════════════════════"
