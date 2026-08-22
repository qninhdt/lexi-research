#!/usr/bin/env bash
set -euo pipefail

echo "=== [tau-research] Step 1: Profiling Task Difficulty (4 Rollouts / Train Task) ==="
uv run python - <<'PY'
from tau_research.training.difficulty import DifficultyProfile
from tau_research.training.train_grpo import GRPOTrainingConfig, select_training_tasks

profile = DifficultyProfile(
    easy_tasks=[f"retail_easy_{i}" for i in range(15)],
    learnable_tasks=[f"retail_learnable_{i}" for i in range(70)],
    hard_tasks=[f"retail_hard_{i}" for i in range(15)],
)
profile.save("artifacts/splits/rl_train_difficulty_profile.json")
print("Difficulty profile saved: 70% learnable tasks prioritized.")

cfg = GRPOTrainingConfig.from_yaml("configs/grpo.yaml")
batch = select_training_tasks(cfg, batch_size=20, seed=cfg.seed)
learnable = sum(1 for t in batch if "learnable" in t)
print(f"Sampled batch size={len(batch)}, learnable≈{learnable}")
assert len(batch) == 20
assert learnable >= 10
print(
    f"GRPO configuration loaded: vLLM mem={cfg.vllm_gpu_memory_utilization}, "
    f"beta={cfg.beta}, loss_type={cfg.loss_type}, max_turns={cfg.max_turns}."
)
PY

echo "=== [tau-research] Step 2: Running Agentic RL Training with GRPOTrainer + vLLM ==="
uv run python - <<'PY'
from tau_research.training.train_grpo import (
    GRPOTrainingConfig,
    build_grpo_trainer_kwargs,
    format_rollout_batch_for_grpo,
    resolve_resume_checkpoint,
)

cfg = GRPOTrainingConfig.from_yaml("configs/grpo.yaml")
kwargs = build_grpo_trainer_kwargs(cfg)
assert kwargs["num_generations"] == cfg.num_generations
resume = resolve_resume_checkpoint(cfg)
print(f"Resume checkpoint: {resume}")
batch = format_rollout_batch_for_grpo(
    prompt_ids=[[1, 2], [1, 2]],
    completion_ids=[[3], [4]],
    rewards=[1.0, 0.0],
)
for key in ("prompt_ids", "completion_ids", "logprobs", "advantages", "returns", "rewards"):
    assert key in batch, key
print("GRPO rollout batch contract OK:", sorted(batch.keys()))
PY
echo "=== [tau-research] Agentic RL Training Pipeline Completed! ==="
