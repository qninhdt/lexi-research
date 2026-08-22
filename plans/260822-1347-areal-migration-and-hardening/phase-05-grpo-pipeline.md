# Phase 05 — Real GRPO Pipeline (TRL rollout_func)

**Status**: pending | **Depends on**: Phase 03, 04 | **Numbers**: theo paper AReaL + ràng buộc L4

## Requirements

- Implement rollout_func thật tương thích TRL đã pin:
  - Input: batch task IDs (difficulty-sampled từ profile thật Phase 03).
  - Với mỗi task: G episode độc lập, mỗi episode chạy `run_episode_rollout` qua env thật.
  - Return đúng schema TRL yêu cầu ở version đã pin (prompt_ids/completion_ids + mask; verify bằng smoke test với installed version — hiện `format_rollout_batch_for_grpo` đang bịa contract).
  - Completion = toàn bộ agent tokens của episode (thinking + actions, multi-turn), mask env/user tokens.
- Sửa mismatch conceptual: `max_completion_length: 1536` là per-turn; nhưng completion của GRPO = cả episode → cần re-size (~4096+) sau khi profile độ dài episode thật. Ghi rõ trong config comment.
- Memory plan cho L4 24GB (từ red-team plan cũ): vllm colocate util 0.20–0.30 + sleep mode, beta=0 (không load reference model), grad ckpt, LoRA r=16.
- Zero-variance resampling: wire `resample_on_zero_variance` + `max_consecutive_zero_variance_batches` vào vòng train thật (hiện chỉ là dataclass field chưa có gì đọc nó).
- W&B: custom metrics `tau/*` từ rollout metadata (success_rate, db_reward, invalid_tool_call_rate...) qua TrajectoryLogger + callback.

## Files

- Modify: `src/tau_grpo...` → chính xác: `src/tau_research/training/train_grpo.py`, `src/typo fix`: `src/tau_research/training/difficulty.py` path
- Create: `src/tau_research/tau/grpo_rollout.py`
- Modify configs/grpo.yaml
- Tests: mở rộng `test_grpo_rollout_adapter.py` (schema khớp TRL đã pin), `test_reward.py`.

## Tests

- Smoke gate (GPU): G=2, 10 train tasks, max_steps=5 — no OOM, reward flow end-to-end, checkpoint save được.
- Unit: zero-variance resampler logic; advantage computation vs numpy reference.

## Risks / Phase Gate

- **Phase gate**: KHÔNG start GRPO full nếu smoke fail bất kỳ mục nào (OOM/deadlock/reward/backward/checkpoint).
- Kỳ vọng kết quả: ΔRL nhỏ (+2 đến +5pp) — bảo vệ bằng paired delta, không kỳ vọng con số lớn như paper 30B.
