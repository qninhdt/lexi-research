#!/usr/bin/env bash
set -euo pipefail

POLICY_MODE="${POLICY_MODE:-dummy}"
MODEL_PATH="${MODEL_PATH:-}"
TASK_IDS="${TASK_IDS:-retail_test_001,retail_test_002}"

echo "=== [tau-research] Running Held-Out Test Evaluation (4 Trials / Task) ==="
echo "Policy mode: ${POLICY_MODE}"

uv run python - <<'PY'
import os
from tau_research.evaluation.evaluate_tau import EvalRunConfig, evaluate_from_config
from tau_research.tau.rollout import MockTauGymEnv
from tau_research.evaluation.metrics import compute_paired_deltas, task_level_scores

cfg = EvalRunConfig.from_yaml("configs/eval.yaml")
print(
    f"Evaluating {cfg.domain} domain on split {cfg.split} "
    f"with {cfg.num_trials} trials/task, max_turns={cfg.max_agent_turns}."
)

policy_mode = os.environ.get("POLICY_MODE", "dummy")
model_path = os.environ.get("MODEL_PATH", "")

class DummyEvalPolicy:
    def __init__(self) -> None:
        self.step = 0

    def generate(self, history):
        self.step += 1
        if self.step % 2 == 1:
            return "<think>Looking up order.</think>\ncall:cancel_order(order_id='100')"
        return (
            "<think>Order cancelled.</think>\n"
            "Your order #100 has been cancelled successfully."
        )

if policy_mode == "dummy" or not model_path:
    policy = DummyEvalPolicy()
else:
    # Placeholder for real HF policy loading when MODEL_PATH is set.
    policy = DummyEvalPolicy()
    print(f"Note: MODEL_PATH={model_path} provided; using DummyEvalPolicy until HF loader is wired.")

task_ids = [t.strip() for t in os.environ.get("TASK_IDS", "retail_test_001,retail_test_002").split(",") if t.strip()]

res = evaluate_from_config(
    cfg,
    task_ids=task_ids,
    policy=policy,
    env_factory=lambda task_id: MockTauGymEnv(task_id=task_id),
)
print("Evaluation Complete!")
print(f"Pass^1 Success Rate: {res['pass_rate']:.2%}")
print(f"95% CI (task-level): [{res['ci_95'][0]:.2%}, {res['ci_95'][1]:.2%}]")

# Smoke paired-delta path with identical checkpoints (delta should be ~0).
task_scores = task_level_scores(res["task_results"])
d_sft, d_rl = compute_paired_deltas(task_scores, task_scores, task_scores)
print(f"Paired deltas sanity (identical): delta_sft={d_sft:.4f}, delta_rl={d_rl:.4f}")
assert abs(d_sft) < 1e-9 and abs(d_rl) < 1e-9
PY

echo "=== [tau-research] Evaluation Report Saved! ==="
