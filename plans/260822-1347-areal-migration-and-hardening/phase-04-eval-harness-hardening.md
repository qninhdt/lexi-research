# Phase 04 — Eval Harness Hardening

**Status**: results file per checkpoint | **Depends on**: Phase 02, 03 | **Blocks**: Phase 05, 06

## Requirements

- **Policy loaders** (hoàn toàn thiếu hiện nay):
  - `HFChatPolicy` (transformers `generate`, decode theo eval.yaml: temperature/top_p/top_k/max_new_tokens, enable_thinking qua template).
  - Optional `VLLMChatPolicy` cho final eval nhanh.
  - Load base / merged-SFT / merged-RL checkpoints bằng path.
- Plumb decoding params từ eval.yaml vào policy (hiện EvalRunConfig chỉ đọc temperature; top_p/top_k/max_generated_tokens_per_turn bị bỏ đi).
- Results file per checkpoint: `artifacts/evaluation/{checkpoint_tag}/eval_results.jsonl` thay vì ghi đè một file duy nhất (mode "w" trong evaluate_task_batch).
- Metrics bổ sung: pass^k (k=2,4) + paired bootstrap CI **cho delta** (compute_paired_deltas hiện chỉ trả mean, không có CI).
- Error taxonomy: map đúng termination_reason thật (`agent_stop`, `max_turns`, `truncation`, `empty_output/action`, `env_truncated`) — hiện taxonomy check các giá trị mà rollout không bao giờ emit.
- Final eval runner: chạy Base/SFT/SFT+RL cùng task IDs + decoding + user sim, sinh `eval_summary.md`.

## Files

- Create: `src/tau_research/evaluation/policies.py`
- HFChatPolicy.generate(history) → raw completion string (thinking + action).
- Modify: `evaluate_tau.py`, `metrics.py`, `error_analysis.py`
- Tests: mở rộng `test_metrics.py` (pass^k, delta CI), `test_error_analysis.py` (map đúng reason thật), `test_policies.py` (mock generate).

## Tests

- Pass^k correctness trên fixture nhỏ (biết trước đáp số).
- Paired-delta CI: hai phân phối biết trước → CI chứa/không chứa 0 như kỳ vọng.

## Risks / Notes

- Eval 40 tasks × 4 trials × 3 checkpoints = 480 episodes × ~8 turns API calls — budget gpt-4.1-mini ok, GPT-4.1 thì tính trước.
- DummyPolicy giữ lại làm smoke mode, không phải default.
