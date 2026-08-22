---
phase: 5
title: "GRPO Training & Difficulty Profiling"
status: completed
priority: P1
dependencies: [3, 4]
---

# Phase 05: GRPO Training & Difficulty Profiling

## Overview
Implement the Agentic Reinforcement Learning pipeline using Hugging Face TRL `GRPOTrainer` with custom `rollout_func` and colocated vLLM inference. Includes an empirical difficulty profiler to filter training tasks (focusing on learnable tasks to maximize reward variance) and a trajectory logger for rich W&B diagnostics.

## Requirements
- Functional:
  - `difficulty.py`: Run $K=4$ rollouts per official Retail training task using the merged SFT model; partition tasks into *easy* ($4/4$), *learnable* ($1/4, 2/4, 3/4$), and *hard* ($0/4$).
  - `train_grpo.py`: Configure TRL `GRPOTrainer` with:
    - Base model: `qwen3.5-2b-tau-retail-sft-merged`
    - Fresh LoRA adapter ($r=16, \alpha=32, \text{dropout}=0.0$)
    <!-- Red Team Finding 1 Fix -->
    - **Strict 24GB L4 Memory Guard**: Pin `vllm_gpu_memory_utilization: 0.20`, `vllm_enable_sleep_mode: true`, `beta: 0.0`, `per_device_train_batch_size: 1`, and `gradient_accumulation_steps: 16` to guarantee zero CUDA OOM during backward optimizer updates.
    - `loss_type="dapo"` (token-normalized sequence loss)
    <!-- Red Team Finding 3 Fix -->
    - **Zero-Variance Batch Resampling**: Monitor `frac_reward_zero_std`; if zero variance persists for consecutive batches, dynamically resample tasks from the learnable pool to maintain non-zero policy gradients.
    - Custom `rollout_func` yielding prompt IDs, completion IDs, logprobs, and environment reward.
  - `trajectory_logger.py`: Log formatted W&B episode tables containing prompt, reasoning tokens, tool calls, DB state changes, and final rewards.
- TDD / Test Suite:
  - `tests/test_difficulty_profiler.py`: Unit test verifying task bucketing logic and weighted sampling probabilities (~70% learnable, 15% easy, 15% hard).
  - `tests/test_grpo_rollout_adapter.py`: Mock test verifying that `rollout_func` passes correctly formatted batches to `GRPOTrainer` and handles zero-variance batch warnings.

## Related Code Files
- Create:
  - `src/tau_research/training/difficulty.py`
  - `src/tau_research/training/train_grpo.py`
  - `src/tau_research/logging/trajectory_logger.py`
  - `tests/test_difficulty_profiler.py`
  - `tests/test_grpo_rollout_adapter.py`
- Modify:
  - `configs/grpo.yaml`

## Implementation Steps
1. **Write TDD Tests First**:
   - Write `tests/test_difficulty_profiler.py` mocking rollout results and checking correct classification of task IDs into JSON split artifacts.
   - Write `tests/test_grpo_rollout_adapter.py` verifying tensor shapes, memory bounds, and zero-variance detection.
2. Implement `difficulty.py`:
   - Iterate through official τ³ Retail train tasks.
   - Run 4 rollouts per task with SFT policy; compute empirical success rate $\hat{p}_i$.
   - Save `artifacts/splits/rl_train_difficulty_profile.json`.
3. Implement `trajectory_logger.py` creating W&B Tables with multi-turn conversation transcripts.
4. Implement `train_grpo.py`:
   - Connect custom `rollout_func` to Tau `AgentGymEnv`.
   - Setup `GRPOConfig` with colocated vLLM settings (`vllm_gpu_memory_utilization: 0.20`, sleep mode enabled) and learning rate $5\times 10^{-6} - 1\times 10^{-5}$.
   - Add stopping / rollback criteria callbacks (monitor `frac_reward_zero_std`, entropy collapse, and dev task reward).

## Success Criteria
- [ ] `uv run pytest tests/test_difficulty_profiler.py tests/test_grpo_rollout_adapter.py` passes.
- [ ] GRPO smoke run (20 steps, $G=2$) executes successfully without OOM or deadlock on Colab L4 GPU.
- [ ] W&B dashboard logs all custom τ metrics (`tau/success_rate`, `tau/db_reward`, `tau/communicate_reward`, `frac_reward_zero_std`).
