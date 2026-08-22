# Phase 03 — Real Env Integration (AgentGymEnv)

**Status**: pending | **Depends on**: Phase 01 | **Blocks**: Phase 04, 05

## Requirements

- `TauEnvFactory` mới (`src/tau_research/tau/env_factory.py`):
  - Load `third_party/tau2-bench/data/tau2/domains/retail/split_tasks.json` → train (74) / test (40) task IDs thật.
  - `create(task_id, split)` → `AgentGymEnv(domain="retail", task_id=..., solo_mode=False, user_llm=..., user_llm_args=...)` từ config (grpo.yaml / eval.yaml).
  - Đọc system prompt/policy từ `info["policy"]`, tools từ `info["tools"]` — đã khớp contract của rollout.py.
- Sửa `rollout.py::run_episode_rollout`:
  - Observation adapter: obs thật là chuỗi formatted `"role: content"` nhiều dòng (không phải user message thuần) — parse hoặc dùng option `all_messages_as_observation=True` rồi rebuild chat messages đúng role; không append nguyên chuỗi vào role "user" như hiện tại.
  - Reward: khi terminated, parse `info["reward_info"]` (JSON) để lấy DB/COMMUNICATE breakdown thay vì fabricate `communicate_success=True`.
  - Giữ truncation = failure policy.
- Difficulty profiler thật (`src/tau_research/training/difficulty.py` thêm runner): chạy SFT-merged checkpoint, 4 rollouts × 74 train tasks qua env thật, classify easy/learnable/hard, ghi đè artifact giả (`retail_easy_0...`) bằng profile thật.
- User simulator: dùng `UserSimulator.build_live_simulator` + config freeze; bỏ default `openai/grok-4.6` lạ trong `user_simulator.py:29` — default theo eval.yaml (`gpt-4.1-mini`) hoặc fail-fast nếu thiếu env var.

## Files

- Create: `src/tau_research/tau/env_factory.py`
- Modify: `src/tau_Research/tau/rollout.py`, `src/tau_research/training/d typo.py` (fix path), `src/tau_research/tau/user_simulator.py`
- Tests: `tests/test_env_factory.py` (mock AgentGymEnv), mở rộng `tests/test_tau_rollout.py` (obs format thật), `tests/test_difficulty_profiler.py` (chạy trên mock env trả reward_info JSON).

## Tests

- Integration test (GPU/API, slow): 1 train task end-to-end reset→generate→step→terminated→reward_info parse được.

## Risks / Notes

- Mỗi `AgentGymEnv.reset()` spawn orchestrator thread + gọi LLM API user-sim → chi phí API thật khi profiling/RL. Budget: profiling = 74×4=296 episodes ≈ vài nghìn lượt gọi mini-model.
- Orchestrator chạy thread daemon; đảm bảo close/join đúng để không leak threads khi rollout hàng loạt.
