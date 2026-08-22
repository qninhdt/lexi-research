#!/usr/bin/env bash
set -euo pipefail

echo "=== [tau-research] Running End-to-End Smoke Test ==="
uv run python -m tau_research.cli smoke --config configs/smoke.yaml
echo "=== [tau-research] Smoke Test Passed! ==="
