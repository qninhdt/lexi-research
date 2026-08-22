"""Multi-turn interactive rollout loop between agent policy and Gym environment."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from tau_research.data.prepare_sft import sanitize_history_for_turn, strip_thinking_tags
from tau_research.tau.action_parser import parse_model_output
from tau_research.tau.env_factory import build_system_prompt
from tau_research.tau.reward import TauReward, calculate_outcome_reward

_ROLE_PREFIX = re.compile(r"^(?:user|assistant|tool|environment):\s?", re.IGNORECASE)


def strip_role_prefix(observation: str) -> str:
    """Converts a formatted gym observation ('user: ...') into plain message content."""
    return _ROLE_PREFIX.sub("", observation.strip(), count=1)


def parse_reward_info(info: dict[str, Any]) -> TauReward | None:
    """Parses the official evaluator's reward_info payload from env step info.

    Returns None when the environment did not produce one (e.g. mock envs).
    """
    raw = info.get("reward_info")
    if not raw:
        return None
    data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    reward = float(data.get("reward") or 0.0)
    breakdown = data.get("reward_breakdown") or {}

    db_val = breakdown.get("DB")
    comm_val = breakdown.get("COMMUNICATE")

    db_check = data.get("db_check") or {}
    if db_val is None and db_check:
        db_val = 1.0 if db_check.get("passed") else 0.0

    comm_checks = data.get("communicate_checks") or []
    if comm_val is None and comm_checks:
        passed = [c.get("passed", False) for c in comm_checks]
        comm_val = 1.0 if passed and all(passed) else 0.0

    partial = float(data.get("partial_action_reward") or 0.0)
    return TauReward(
        reward=reward,
        db_reward=float(db_val) if db_val is not None else (1.0 if reward > 0 else 0.0),
        communicate_reward=(
            float(comm_val) if comm_val is not None else (1.0 if reward > 0 else 0.0)
        ),
        partial_action_reward=partial,
        is_success=reward >= 1.0,
    )


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
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Executes a full multi-turn rollout until environment termination or truncation.

    When the environment exposes a domain policy (real AgentGymEnv), the system
    prompt defaults to the training-format instructions+policy wrapper so
    inference prompts stay in-distribution with SFT prompts.
    """
    obs, info = env.reset()
    resolved_system_prompt = system_prompt
    if resolved_system_prompt is None:
        env_policy = info.get("policy") if isinstance(info, dict) else None
        resolved_system_prompt = (
            build_system_prompt(str(env_policy))
            if env_policy
            else "You are a helpful customer service assistant for Retail operations."
        )

    history: list[dict[str, Any]] = [
        {"role": "system", "content": resolved_system_prompt},
        {"role": "user", "content": strip_role_prefix(str(obs))},
    ]

    terminated = False
    truncated = False
    turn_count = 0
    last_step_reward = 0.0
    step_rewards: list[float] = []
    termination_reason: str | None = None
    last_action = ""
    final_reward = calculate_outcome_reward(False, False)

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
            history.append({"role": "tool", "content": strip_role_prefix(str(obs))})
        else:
            history.append({"role": "user", "content": strip_role_prefix(str(obs))})

        hit_max_turns = turn_count >= max_turns and not terminated
        if terminated:
            # Prefer the official evaluator's breakdown when present.
            official = parse_reward_info(step_info) if isinstance(step_info, dict) else None
            if official is not None:
                final_reward = official
            else:
                final_reward = calculate_outcome_reward(
                    db_success=last_step_reward == 1.0,
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
