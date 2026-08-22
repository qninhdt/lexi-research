#!/usr/bin/env bash
# Final held-out evaluation: Base vs SFT vs SFT+RL on official retail test split.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
SFT_MERGED="${SFT_MERGED:-artifacts/models/qwen3.5-2b-tau-retail-sft-merged}"
RL_MERGED="${RL_MERGED:-artifacts/models/qwen3.5-2b-tau-retail-grpo-merged}"
POLICY="${POLICY:-hf}"   # hf | vllm

run_eval () {
    local tag="$1" path="$2"
    echo "=== [tau-research] Evaluating ${tag}: ${path} ==="
    uv run tau-research evaluate \
        --config configs/eval.yaml \
        --model-path "${path}" \
        --tag "${tag}" \
        --policy "${POLICY}"
}

run_eval base "${BASE_MODEL}"
run_eval sft "${SFT_MERGED}"
run_eval rl "${RL_MERGED}"

echo "=== [tau-research] Comparing checkpoints (paired bootstrap deltas) ==="
uv run python - <<'PY'
import json
from pathlib import Path

from tau_research.evaluation.metrics import paired_bootstrap_delta, task_level_scores

def load(tag: str) -> dict[str, float]:
    path = Path(f"artifacts/evaluation/{tag}/eval_results.jsonl")
    per_task: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        per_task.setdefault(rec["task_id"], []).append(rec["reward"])
    return task_level_scores(per_task)

base, sft, rl = load("base"), load("sft"), load("rl")
d_sft = paired_bootstrap_delta(base, sft)
d_rl = paired_bootstrap_delta(sft, rl)
print(f"Delta SFT - Base : {d_sft['delta']:+.4f}  95% CI [{d_sft['ci_low']:+.4f}, {d_sft['ci_high']:+.4f}]")
print(f"Delta RL  - SFT  : {d_rl['delta']:+.4f}  95% CI [{d_rl['ci_low']:+.4f}, {d_rl['ci_high']:+.4f}]")

summary = {"delta_sft": d_sft, "delta_rl": d_rl}
Path("artifacts/evaluation/paired_deltas.json").write_text(json.dumps(summary, indent=2))
print("Saved artifacts/evaluation/paired_deltas.json")
PY

echo "=== [tau-research] Final evaluation complete ==="
