"""Multi-turn interactive rollout loop between agent policy and Gym environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tau_research.data.prepare_sft import sanitize_history_for_turn, strip_thinking_tags
from tau_research.tau.action_parser import parse_model_output
from tau_research.tau.reward import TauReward, calculate_outcome_reward


class MockTauGymEnv:
    """Mock Gym Environment for unit testing and offline verification."""

    def __init__(self, task_id: str = "mock_001") -> None:
        self.task_id = task_id
        self._initial_db: dict[str, Any] = {"orders": {"100": "pending"}}
        self.db_state: dict[str, Any] = deepcopy(self._initial_db)
        self.current_step = 0

    def reset(self) -> tuple[str, dict[str, Any]]:
        self.db_state = deepcopy(self._initial_db)
        self.current_step = 0
        obs = "Hello, I want to cancel my order #100."
        info = {
            "task_id": self.task_id,
            "policy": "You are a customer service assistant.",
            "tools": ["get_order", "cancel_order"],
        }
        return obs, info

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Accept plain text or tau2-style functional/JSON tool-call strings."""
        self.current_step += 1
        action_text = action if isinstance(action, str) else str(action)
        action_text = action_text.strip()

        # Match cancel by tool name only (not by bare order_id= which also appears on get_order).
        is_cancel = "cancel_order" in action_text
        is_lookup = "get_order" in action_text and not is_cancel

        if is_cancel:
            self.db_state["orders"]["100"] = "cancelled"
            obs = '{"status": "cancelled", "order_id": "100"}'
            return obs, 0.0, False, False, {"step_reward": 0.0}

        if is_lookup:
            status = self.db_state["orders"]["100"]
            obs = f'{{"order_id": "100", "status": "{status}"}}'
            return obs, 0.0, False, False, {"step_reward": 0.0}

        # Final plain-text answer ends the episode.
        obs = "Thank you for the help!"
        terminated = True
        reward = 1.0 if self.db_state["orders"]["100"] == "cancelled" else 0.0
        return obs, reward, terminated, False, {"step_reward": reward}


def run_episode_rollout(
    env: Any,
    policy: Any,
    max_turns: int = 8,
    system_prompt: str = "You are a helpful customer service assistant for Retail operations.",
) -> dict[str, Any]:
    """Executes a full multi-turn rollout until environment termination or truncation."""
    obs, info = env.reset()
    history: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs},
    ]

    terminated = False
    truncated = False
    turn_count = 0
    last_step_reward = 0.0
    step_rewards: list[float] = []
    termination_reason: str | None = None
    last_action = ""
    final_reward: TauReward = calculate_outcome_reward(False, False)

    while not (terminated or truncated) and turn_count < max_turns:
        turn_count += 1
        prompt_history = sanitize_history_for_turn(history)

        raw_output = policy.generate(prompt_history)
        parsed = parse_model_output(raw_output)
        last_action = raw_output

        if parsed.is_truncated:
            truncated = True
            termination_reason = parsed.termination_reason or "truncation"
            # Truncation is a failed episode: zero outcome reward.
            final_reward = calculate_outcome_reward(
                db_success=False,
                communicate_success=False,
            )
            break

        # tau2 Gym expects functional/JSON/plain strings — never raw call: prefixes.
        action_payload = parsed.to_env_action()
        obs, reward_val, terminated, env_truncated, step_info = env.step(action_payload)
        truncated = bool(env_truncated)
        last_step_reward = float(reward_val)
        step_rewards.append(last_step_reward)

        cleaned_assistant_output = strip_thinking_tags(raw_output)
        # Prefer functional form in history so subsequent turns stay tau2-compatible.
        history_assistant = action_payload if parsed.is_tool_call else cleaned_assistant_output
        history.append({"role": "assistant", "content": history_assistant})
        if parsed.is_tool_call:
            history.append({"role": "tool", "content": obs})
        else:
            history.append({"role": "user", "content": obs})

        hit_max_turns = turn_count >= max_turns and not terminated
        if terminated:
            final_reward = calculate_outcome_reward(
                db_success=reward_val == 1.0,
                communicate_success=True,
            )
            termination_reason = "agent_stop"
        elif hit_max_turns:
            truncated = True
            termination_reason = "max_turns"
            final_reward = calculate_outcome_reward(
                db_success=False,
                communicate_success=False,
            )
        elif truncated:
            termination_reason = "env_truncated"
            final_reward = calculate_outcome_reward(
                db_success=False,
                communicate_success=False,
            )

    return {
        "task_id": getattr(env, "task_id", "unknown"),
        "num_turns": turn_count,
        "terminated": terminated,
        "truncated": truncated,
        "reward": final_reward,
        "step_rewards": step_rewards,
        "last_step_reward": last_step_reward,
        "termination_reason": termination_reason,
        "last_action": last_action,
        "history": history,
    }
