#!/usr/bin/env bash
# Run the QLoRA training pipeline on a Colab GPU through google-colab-cli.
#
# Usage:
#   bash run_colab_train.sh L4 qwen
#   TRAIN_PATH=data/clean/train.parquet MAX_STEPS=0 \
#     OUTPUT_DIR=models/qwen35 bash run_colab_train.sh L4 qwen
#   TRAIN_PATH=data/clean/train.parquet MAX_STEPS=0 \
#     OUTPUT_DIR=models/gemma4 bash run_colab_train.sh L4 gemma
#
# Requires: google-colab-cli (`uv tool install google-colab-cli`)

set -euo pipefail

GPU="${1:-L4}"
MODEL="${2:-qwen}"
GPU_KEY="${GPU^^}"
MODEL_KEY="${MODEL,,}"
SESSION="${SESSION:-lexi-train-${MODEL_KEY}}"
TRAIN_PATH="${TRAIN_PATH:-ops/fixtures/smoke_50.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/smoke_test-${MODEL_KEY}}"
MAX_STEPS="${MAX_STEPS:-2}"
EPOCHS="${EPOCHS:-1}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-7200}"
# Where the remote VM keeps its Hugging Face cache. Empty means the VM-local
# default, so each session re-downloads the checkpoint (~40 s for a 4B model).
# Point it at a mounted Drive path — e.g.
#   colab drivemount -s "$SESSION"
#   HF_HOME=/content/drive/MyDrive/hf-cache bash run_colab_train.sh L4 qwen
# — to keep the download between sessions.
HF_HOME="${HF_HOME:-}"
# `auto` continues from the newest checkpoint in OUTPUT_DIR, `none` starts over.
# This matters because a full run does not fit in one Colab session: 16,320 rows
# x 2 epochs at ~2.3 s/sequence is roughly 21 hours, and a session is killed
# well before that. Checkpoints land every `train.save_steps` optimiser steps.
RESUME="${RESUME:-auto}"
case "${RESUME}" in
  auto|none) ;;
  *[!A-Za-z0-9_./-]*|"")
    echo "RESUME must be 'auto', 'none', or a checkpoint path." >&2
    exit 2
    ;;
esac
# Where the *remote* run writes checkpoints. Unset, they go to VM-local disk and
# exist only until the VM does — a session killed at hour 6 of a 21-hour run
# takes them with it. Point this at a mounted Drive directory and the Trainer
# writes each `train.save_steps` checkpoint straight to storage that outlives the
# VM, so a relaunch with RESUME=auto finds them:
#
#   colab drivemount -s "$SESSION"
#   REMOTE_OUTPUT_DIR=/content/drive/MyDrive/lexi-runs/qwen35 \
#     HF_HOME=/content/drive/MyDrive/hf-cache bash run_colab_train.sh L4 qwen
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-}"
case "${REMOTE_OUTPUT_DIR}" in
  "") ;;
  *[!A-Za-z0-9_./-]*)
    echo "REMOTE_OUTPUT_DIR must contain only letters, digits, '_', '.', '/', and '-'." >&2
    exit 2
    ;;
esac

# Per-device batch size is a per-model memory fact on a 22 GiB L4, not a
# preference. Both pairs multiply to the same effective batch of 32, so the
# learning rate carries over between them. Measured over 64 sequences:
#
#   Qwen3.5-4B     batch 4 -> 171 s,  9.5 GiB peak
#   Gemma 4 E4B-it batch 1 -> 220 s, 20.6 GiB peak; batch 4 runs out of memory
#
# Override BATCH_SIZE/GRAD_ACCUM together, or the effective batch moves with it.
case "${MODEL_KEY}" in
  qwen|qwen3.5|qwen3.5-4b)
    MODEL_KEY="qwen"
    MODEL_ID="Qwen/Qwen3.5-4B"
    DEFAULT_BATCH_SIZE=4
    DEFAULT_GRAD_ACCUM=8
    ;;
  gemma|gemma4|gemma-4-e4b|gemma-4-e4b-it)
    MODEL_KEY="gemma"
    MODEL_ID="google/gemma-4-E4B-it"
    # E4B leaves under 1.5 GiB free at batch 1; four sequences OOM.
    DEFAULT_BATCH_SIZE=1
    DEFAULT_GRAD_ACCUM=32
    ;;
  *)
    echo "Unsupported model '${MODEL}'. Use 'qwen' or 'gemma'." >&2
    exit 2
    ;;
esac

BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
GRAD_ACCUM="${GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}"

# Extra `--override key.path=value` pairs, space-separated. Stage A needs
# `train.task=corrector`, which selects the correction-only collator and prompt:
#
#   TRAIN_PATH=data/gec/train.parquet MAX_STEPS=0 \
#     EXTRA_OVERRIDES="train.task=corrector train.rubric=terse train.thinking=off" \
#     bash run_colab_train.sh L4 qwen
#
# Restricted to the same character class as the other embedded values, because
# these land inside a remote Python snippet and a shell metacharacter here would
# become code there.
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
case "${EXTRA_OVERRIDES}" in
  *[!A-Za-z0-9_.=\ -]*)
    echo "EXTRA_OVERRIDES may contain only letters, digits, '_', '.', '=', '-', and spaces." >&2
    exit 2
    ;;
esac

case "${GPU_KEY}" in
  L4) REQUIREMENTS_FILE="requirements-colab-l4.txt" ;;
  *) REQUIREMENTS_FILE="requirements-colab.txt" ;;
esac

# Values are embedded in the remote Python snippets below. Keep paths and
# scalar overrides deliberately shell-safe so a typo cannot become Python code.
for value_name in TRAIN_PATH OUTPUT_DIR; do
  value="${!value_name}"
  case "${value}" in
    *[!A-Za-z0-9_./-]*|"")
      echo "${value_name} must contain only letters, digits, '_', '.', '/', and '-'." >&2
      exit 2
      ;;
  esac
done
# HF_HOME is optional, but it is interpolated into the remote snippet like the
# paths above, so when set it gets the same treatment.
case "${HF_HOME}" in
  "") ;;
  *[!A-Za-z0-9_./-]*)
    echo "HF_HOME must contain only letters, digits, '_', '.', '/', and '-'." >&2
    exit 2
    ;;
esac
case "${MAX_STEPS}" in *[!0-9]*|"") echo "MAX_STEPS must be a non-negative integer." >&2; exit 2 ;; esac
case "${EPOCHS}" in *[!0-9]*|"") echo "EPOCHS must be a non-negative integer." >&2; exit 2 ;; esac
case "${BATCH_SIZE}" in *[!0-9]*|""|0) echo "BATCH_SIZE must be a positive integer." >&2; exit 2 ;; esac
case "${GRAD_ACCUM}" in *[!0-9]*|""|0) echo "GRAD_ACCUM must be a positive integer." >&2; exit 2 ;; esac

echo "═══════════════════════════════════════════════════════"
echo "🚀  Lexi-Research — Colab Training"
echo "    GPU: ${GPU}   Model: ${MODEL_ID}   Session: ${SESSION}"
echo "    Data: ${TRAIN_PATH}   Steps: ${MAX_STEPS}   Deps: ${REQUIREMENTS_FILE}"
echo "    Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} accum = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "    Resume: ${RESUME}   Remote output: ${REMOTE_OUTPUT_DIR:-<VM-local, lost if the VM dies>}"
echo "═══════════════════════════════════════════════════════"

# The VM is stopped on success, and deliberately left running on failure.
#
# A long run's checkpoints live on VM-local disk until step 5 downloads them, so
# tearing the VM down on *any* exit — a failed step, or a Ctrl-C — destroys the
# progress the user most wants to keep. Leaving it up costs compute units and
# says so; the alternative silently discards hours of training.
#
# KEEP_VM=1 keeps it up even after a successful run, for inspecting artifacts.
cleanup() {
  status=$?
  if [ "${status}" -eq 0 ] && [ "${KEEP_VM:-0}" != "1" ]; then
    colab stop -s "${SESSION}" >/dev/null 2>&1 || true
    return
  fi
  if [ "${status}" -ne 0 ]; then
    echo "" >&2
    echo "⚠️  Exited with status ${status}. Session '${SESSION}' is still RUNNING so" >&2
    echo "    its checkpoints survive. It consumes compute units until stopped." >&2
    echo "    Resume:   RESUME=auto bash $0 ${GPU} ${MODEL_KEY}" >&2
    echo "    Inspect:  colab ls -s ${SESSION} /content/lexi-research/${OUTPUT_DIR}" >&2
    echo "    Stop:     colab stop -s ${SESSION}" >&2
  else
    echo ""
    echo "KEEP_VM=1 — session '${SESSION}' left running. Stop it with:"
    echo "    colab stop -s ${SESSION}"
  fi
}
trap cleanup EXIT

# ── 1. Provision ──────────────────────────────────────────
echo ""
echo "--- [1/5] Provisioning Colab VM (GPU=${GPU}) ---"
# Reuse a session that is still alive. A failed run leaves its VM up (see
# cleanup above) precisely so a resume can pick up the checkpoints already on
# its disk; provisioning a fresh VM instead would start from an empty directory
# and `--resume auto` would silently begin at step 0.
#
# The exit status of `colab status` cannot be used for this: it returns 0 for a
# session that does not exist, printing "not found" to stdout. Testing it alone
# sent every first run down the reuse path, which skipped provisioning and then
# failed at the GPU check against a VM that was never created.
if colab status -s "${SESSION}" 2>&1 | grep -qv "not found"; then
  echo "Reusing live session '${SESSION}' and whatever it has already written."
  REUSED_SESSION=1
else
  colab new -s "${SESSION}" --gpu "${GPU}"
  REUSED_SESSION=0
fi

# ── 2. GPU check ──────────────────────────────────────────
echo ""
echo "--- [2/5] Verifying GPU ---"
echo "import subprocess; print(subprocess.check_output(['nvidia-smi']).decode())" \
  | colab exec -s "${SESSION}" --timeout 120

# ── 3. Clone repo & install deps ─────────────────────────
echo ""
echo "--- [3/5] Cloning repo & installing dependencies ---"
cat <<PYSETUP | colab exec -s "${SESSION}" --timeout "${EXEC_TIMEOUT}"
import os
import subprocess

# Idempotent: a resumed run reuses a live VM whose repo and wheels are already
# there, and re-cloning into an existing directory fails. Updating in place also
# keeps everything the previous attempt wrote, which is the point of resuming.
repo = "/content/lexi-research"
if os.path.isdir(repo + "/.git"):
    print(">>> repo present; fetching instead of cloning", flush=True)
    cmds = [f"git -C {repo} fetch --depth 1 origin && git -C {repo} reset --hard origin/HEAD"]
else:
    cmds = [f"git clone --depth 1 https://github.com/qninhdt/lexi-research.git {repo}"]
cmds += [
    "python -m pip install --disable-pip-version-check -q -r /content/lexi-research/${REQUIREMENTS_FILE}",
    "python -m pip install --disable-pip-version-check -q pydantic jinja2 openai httpx fastapi uvicorn",
]
for cmd in cmds:
    print(f"\n>>> {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

print("\n>>> checking fused kernels", flush=True)
import torch
print(f"torch={torch.__version__}; cuda={torch.version.cuda}; gpu={torch.cuda.get_device_name(0)}")
try:
    import fla  # noqa: F401
    print("flash-linear-attention: enabled")
except Exception as exc:
    print(f"flash-linear-attention: unavailable ({type(exc).__name__}: {exc})")
try:
    import causal_conv1d  # noqa: F401
    print("causal-conv1d: enabled")
except Exception as exc:
    print(f"causal-conv1d: unavailable ({type(exc).__name__}: {exc})")
print("\n✅ Environment ready", flush=True)
PYSETUP

# ── 4. Train ─────────────────────────────────────────────
echo ""
echo "--- [4/5] Running QLoRA training ---"
cat <<PYTRAIN | colab exec -s "${SESSION}" --timeout "${EXEC_TIMEOUT}"
import os
import subprocess
import sys

os.chdir("/content/lexi-research")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# A fresh Colab VM starts with no model cache, so the checkpoint download is
# paid once per session. The Rust downloader makes that transfer parallel; the
# 4B Qwen3.5 checkpoint measured 39.7 s with it enabled. Loading an already
# cached checkpoint takes about 13 s and is not what this affects.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
if "${HF_HOME}":
    # Pointing this at a mounted Drive path keeps the download across sessions,
    # turning a repeated 40 s into a one-off. Unset, the VM-local cache is used
    # and the download repeats per session.
    os.environ["HF_HOME"] = "${HF_HOME}"

cmd = [
    sys.executable, "-m", "lexi_research.cli", "train", "sft",
    "--train", "${TRAIN_PATH}",
    # A Drive-backed output directory keeps checkpoints when the VM goes away.
    "--output", "${REMOTE_OUTPUT_DIR}" or "${OUTPUT_DIR}",
    "--override", "train.base_model=${MODEL_ID}",
    "--override", "train.epochs=${EPOCHS}",
    "--override", "train.max_steps=${MAX_STEPS}",
    "--override", "train.text_only=true",
    "--override", "train.attn_implementation=sdpa",
    "--override", "train.selective_logits=true",
    "--override", "train.per_device_batch_size=${BATCH_SIZE}",
    "--override", "train.grad_accum=${GRAD_ACCUM}",
    "--resume", "${RESUME}",
]

for pair in "${EXTRA_OVERRIDES}".split():
    cmd += ["--override", pair]

if "${GPU_KEY}" == "L4":
    # L4 fits this 4B hybrid model with recomputation; without it, the
    # roughly 1.5k-token smoke examples exhaust the 22 GiB device.
    cmd += [
        "--override", "train.gradient_checkpointing=true",
        "--override", "train.tf32=true",
        "--override", "train.bnb_4bit_use_double_quant=false",
        "--override", "train.dataloader_num_workers=2",
        "--override", "train.dataloader_persistent_workers=true",
        "--override", "train.dataloader_prefetch_factor=2",
    ]

print(f">>> {' '.join(cmd)}", flush=True)
result = subprocess.run(cmd)
if result.returncode:
    print("❌ Training failed!", flush=True)
    sys.exit(result.returncode)
print("\n✅ Training complete!", flush=True)
PYTRAIN

# ── 5. Download & cleanup ────────────────────────────────
echo ""
echo "--- [5/5] Downloading results & stopping VM ---"
mkdir -p "${OUTPUT_DIR}"
# An absolute REMOTE_OUTPUT_DIR (a Drive mount) is already outside the VM, so it
# is fetched from where the run actually wrote; otherwise it is relative to the
# repo checkout on VM-local disk.
case "${REMOTE_OUTPUT_DIR}" in
  /*) REMOTE_FETCH_PATH="${REMOTE_OUTPUT_DIR}" ;;
  "") REMOTE_FETCH_PATH="/content/lexi-research/${OUTPUT_DIR}" ;;
  *)  REMOTE_FETCH_PATH="/content/lexi-research/${REMOTE_OUTPUT_DIR}" ;;
esac
colab download -s "${SESSION}" \
  "${REMOTE_FETCH_PATH}" "${OUTPUT_DIR}" \
  || echo "⚠️  Download failed or no output files"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🎉  Done! Check ${OUTPUT_DIR}/ for adapter weights."
echo "═══════════════════════════════════════════════════════"
