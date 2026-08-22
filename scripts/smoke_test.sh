#!/usr/bin/env bash
# CPU smoke gate: converter, render, parser round-trip, config wiring.
set -euo pipefail

echo "=== [tau-research] Smoke: unit suite ==="
uv run pytest -q

echo "=== [tau-research] Smoke: AReaL conversion on fixture ==="
uv run tau-research convert-areal --input tests/fixtures/areal_sample.jsonl --out-dir /tmp/tau_smoke_data

echo "=== [tau-research] Smoke: SFT dry-run render (fixture data in /tmp) ==="
python3 - <<'PYCFG'
import yaml

cfg = yaml.safe_load(open("configs/sft.yaml"))
cfg["dataset"]["train_path"] = "/tmp/tau_smoke_data/areal_sft_train.json"
cfg["dataset"]["val_path"] = "/tmp/tau_smoke_data/areal_sft_val.json"
yaml.safe_dump(cfg, open("/tmp/tau_smoke_sft.yaml", "w"))
PYCFG
uv run tau-research train-sft --config /tmp/tau_smoke_sft.yaml --dry-run

echo "=== [tau-research] Smoke: GRPO dry-run wiring ==="
uv run tau-research train-grpo --config configs/grpo.yaml --dry-run

echo "=== [tau-research] Smoke passed ==="
