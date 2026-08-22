#!/usr/bin/env bash
# Full SFT stage: convert AReaL data (if needed) then LoRA reasoning SFT.
set -euo pipefail

AREAL_JSONL="${AREAL_JSONL:-data/areal/tau2_sft_train.jsonl}"

if [ ! -f "artifacts/data/areal_sft_train.json" ]; then
    echo "=== [tau-research] Converting AReaL SFT data from ${AREAL_JSONL} ==="
    uv run tau-research convert-areal --input "${AREAL_JSONL}" --out-dir artifacts/data
fi

echo "=== [tau-research] Dry-run render check ==="
uv run tau-research train-sft --config configs/sft.yaml --dry-run

echo "=== [tau-research] Running LoRA reasoning SFT ==="
uv run tau-research train-sft --config configs/sft.yaml

echo "=== [tau-research] SFT stage complete: artifacts/models/qwen3.5-2b-tau-retail-sft-merged ==="
