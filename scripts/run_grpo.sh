#!/usr/bin/env bash
# RL stage: difficulty profiling on the SFT policy, then online GRPO.
set -euo pipefail

SFT_MERGED="${SFT_MERGED:-artifacts/models/qwen3.5-2b-tau-retail-sft-merged}"

echo "=== [tau-research] Difficulty profiling (4 rollouts x 74 official train tasks) ==="
uv run tau-research profile-difficulty \
    --model-path "${SFT_MERGED}" \
    --output artifacts/splits/rl_train_difficulty_profile.json

echo "=== [tau-research] Online multi-turn GRPO ==="
uv run tau-research train-grpo --config configs/grpo.yaml

echo "=== [tau-research] RL stage complete ==="
