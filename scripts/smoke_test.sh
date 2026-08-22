#!/usr/bin/env bash
# CPU smoke gate: converter, render, parser round-trip, config wiring.
set -euo pipefail

echo "=== [tau-research] Smoke: unit suite ==="
uv run pytest -q

echo "=== [tau-research] Smoke: AReaL conversion on fixture ==="
uv run tau-research convert-areal --input tests/fixtures/areal_sample.jsonl --out-dir /tmp/tau_smoke_data

echo "=== [tau-research] Smoke: SFT dry-run render ==="
uv run tau-research train-sft --config configs/sft.yaml --dry-run

echo "=== [tau-research] Smoke: GRPO dry-run wiring ==="
uv run tau-research train-grpo --config configs/grpo.yaml --dry-run

echo "=== [tau-research] Smoke passed ==="
