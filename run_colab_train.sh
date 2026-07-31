#!/usr/bin/env bash
# Smoke-test the QLoRA training pipeline on a free Colab GPU.
#
# Uploads the local project directly (no git clone needed).
#
# Usage:  bash run_colab_train.sh [GPU]
#         bash run_colab_train.sh T4       # default
#         bash run_colab_train.sh A100
#
# Requires: google-colab-cli (`uv tool install google-colab-cli`)

set -euo pipefail

GPU="${1:-T4}"
SESSION="lexi-train"
REMOTE_DIR="/content/lexi-research"

echo "═══════════════════════════════════════════════════════"
echo "🚀  Lexi-Research — Colab Training Smoke Test"
echo "    GPU: $GPU   Session: $SESSION"
echo "═══════════════════════════════════════════════════════"

# ── 1. Provision ──────────────────────────────────────────
echo ""
echo "--- [1/6] Provisioning Colab VM (GPU=$GPU) ---"
colab new -s "$SESSION" --gpu "$GPU"

# ── 2. GPU check ──────────────────────────────────────────
echo ""
echo "--- [2/6] Verifying GPU ---"
echo "import subprocess; print(subprocess.check_output(['nvidia-smi']).decode())" \
  | colab exec -s "$SESSION"

# ── 3. Upload project files ──────────────────────────────
echo ""
echo "--- [3/6] Uploading project files ---"

# Create remote directory structure first
cat <<'PYINIT' | colab exec -s "$SESSION"
import os
dirs = [
    "/content/lexi-research/lexi_research/format",
    "/content/lexi-research/lexi_research/teacher/prompts",
    "/content/lexi-research/lexi_research/train",
    "/content/lexi-research/lexi_research/data",
    "/content/lexi-research/lexi_research/eval",
    "/content/lexi-research/data",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  mkdir {d}")
print("✅ Directories created")
PYINIT

# Upload all necessary source files
upload() { colab upload -s "$SESSION" "$1" "$REMOTE_DIR/$1"; }

# Core package
upload lexi_research/__init__.py
upload lexi_research/format/__init__.py
upload lexi_research/format/bands.py
upload lexi_research/format/parser.py
upload lexi_research/format/tags.py
upload lexi_research/format/validate.py
upload lexi_research/teacher/__init__.py
upload lexi_research/teacher/schemas.py
upload lexi_research/teacher/registry.py
upload lexi_research/teacher/client.py
upload lexi_research/teacher/cache.py
upload lexi_research/teacher/probe.py
upload lexi_research/teacher/prompts/grader_system.jinja
upload lexi_research/teacher/prompts/grader_user.jinja
upload lexi_research/teacher/prompts/diversify_system.jinja
upload lexi_research/teacher/prompts/diversify_user.jinja
upload lexi_research/train/__init__.py
upload lexi_research/train/cli.py
upload lexi_research/train/dataset.py
upload lexi_research/train/trainer.py
upload lexi_research/data/__init__.py 2>/dev/null || true
upload lexi_research/eval/__init__.py

# Config & data
upload band_config.json
upload requirements-colab.txt
upload data/sample_train.parquet

echo "✅ All files uploaded"

# ── 4. Install deps ──────────────────────────────────────
echo ""
echo "--- [4/6] Installing dependencies ---"
cat <<'PYSETUP' | colab exec -s "$SESSION"
import subprocess, sys
cmds = [
    "pip install -q -r /content/lexi-research/requirements-colab.txt",
    "pip install -q pyarrow pydantic jinja2 openai httpx fastapi uvicorn",
]
for cmd in cmds:
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)
print("\n✅ Dependencies installed")
PYSETUP

# ── 5. Train ─────────────────────────────────────────────
echo ""
echo "--- [5/6] Running QLoRA training (1 epoch, 32 samples) ---"
cat <<'PYTRAIN' | colab exec -s "$SESSION"
import subprocess, sys, os
os.chdir("/content/lexi-research")
os.environ["WANDB_DISABLED"] = "true"

cmd = [
    sys.executable, "-m", "lexi_research.train.cli",
    "--train", "data/sample_train.parquet",
    "--output", "models/smoke_test",
    "--epochs", "1",
]
print(f">>> {' '.join(cmd)}")
result = subprocess.run(cmd)
if result.returncode != 0:
    print("❌ Training failed!")
    sys.exit(result.returncode)
print("✅ Training complete!")
PYTRAIN

# ── 6. Download & cleanup ────────────────────────────────
echo ""
echo "--- [6/6] Downloading results & stopping VM ---"
mkdir -p models/smoke_test
colab download -s "$SESSION" /content/lexi-research/models/smoke_test models/smoke_test \
  || echo "⚠️  Download failed or no output files"
colab stop -s "$SESSION"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🎉  Done! Check models/smoke_test/ for adapter weights."
echo "═══════════════════════════════════════════════════════"
