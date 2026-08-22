---
phase: 3
title: "Tau Env, Action Parser & Rollout Adapter"
status: completed
priority: P1
dependencies: [1]
---

# Phase 03: Tau Env, Action Parser & Rollout Adapter

## Overview
Implement the environment integration and action execution layer connecting `sierra-research/tau2-bench` v1.0.1 `AgentGymEnv` with the Qwen3.5 policy model. This module handles action string parsing (separating thinking blocks from tool calls or direct user messages), environment stepping, reward extraction ($R = R_{\text{DB}} \times R_{\text{COMM}}$), and frozen user simulation.

## Requirements
- Functional:
  - `action_parser.py`: Parse raw model output into structured `Action` objects:
    - Extract `<think>...</think>` content.
    - Parse tool call name and JSON arguments (or plain text communication).
    - Handle JSON malformed syntax gracefully with fallback recovery.
    - <!-- Red Team Finding 2 Fix --> **Truncation & Loop Fallback**: Detect unclosed `<think>` tags and truncated outputs when `max_generated_tokens_per_turn` is reached. Return a standardized `TruncatedAction` that cleanly terminates the episode and records a truncation event without crashing the rollout loop.
  - `user_simulator.py`: Wrapper for frozen user simulator (`solo_mode=False`) maintaining consistent persona across rollouts.
  - `reward.py`: Extract outcome-based binary reward ($R = R_{\text{DB}} \times R_{\text{COMM}}$) and diagnostic metrics (`partial_action_reward`, DB state delta).
  - `rollout.py`: Multi-turn conversational interaction loop executing an episode until termination/truncation, maintaining sanitized turn history (stripping previous `<think>` tokens).
    - <!-- Red Team Finding 5 Fix --> **Atomic DB State Isolation**: Ensure `env.reset()` completely restores the SQLite DB state from a clean fixture for every task and trial, preventing state leakage across episodes.
- TDD / Test Suite:
  - `tests/test_action_parser.py`: Unit tests with varied model completions (valid tool call, plain communicate, markdown code block tool call, malformed JSON arguments, and truncated unclosed reasoning traces).
  - `tests/test_reward.py`: Unit tests asserting binary outcome reward calculation and diagnostic metrics.
  - `tests/test_tau_rollout.py`: Mock Gym environment tests verifying multi-turn episode stepping, termination detection, history pruning, and database state reset purity.

## Related Code Files
- Create:
  - `src/tau_research/tau/__init__.py`
  - `src/tau_research/tau/action_parser.py`
  - `src/tau_research/tau/user_simulator.py`
  - `src/tau_research/tau/reward.py`
  - `src/tau_research/tau/rollout.py`
  - `tests/test_action_parser.py`
  - `tests/test_reward.py`
  - `tests/test_tau_rollout.py`
- Modify:
  - `configs/eval.yaml`

## Implementation Steps
1. **Write TDD Tests First**:
   - Write `tests/test_action_parser.py` testing edge cases: tool calls embedded in thinking tags, unclosed thinking tags, markdown JSON blocks, invalid JSON strings, and mid-token truncations.
   - Write `tests/test_reward.py` checking combinations of $R_{\text{DB}} \in \{0, 1\}$ and $R_{\text{COMM}} \in \{0, 1\}$.
   - Write `tests/test_tau_rollout.py` creating a mock Gym environment to verify the multi-turn interaction protocol and state reset purity after mutations.
2. Implement `action_parser.py`:
   - Regex / token-based extraction for `<think>` reasoning block.
   - Tool calling parsing matching Qwen3.5 tool call format.
   - Robust truncation fallback yielding `TruncatedAction(reason="max_tokens_exceeded")`.
3. Implement `reward.py` extracting official Tau reward and diagnostic sub-rewards.
4. Implement `user_simulator.py` managing user LLM API configuration with fallback handling.
5. Implement `rollout.py` orchestrating the full episode loop with max turn bounds and step-level history accumulation.

## Success Criteria
- [ ] `uv run pytest tests/test_action_parser.py tests/test_reward.py tests/test_tau_rollout.py` passes 100%.
- [ ] Action parser correctly identifies tool calls, plain messages, and truncated reasoning traces in < 1ms per completion.
- [ ] Mock rollouts correctly complete episodes and return trajectory dictionaries with token IDs and reward stats.
