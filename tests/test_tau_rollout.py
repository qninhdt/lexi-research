"""Tests for mock Gym environment rollout loop, history stripping, and DB reset isolation."""

from typing import Any

from tau_research.tau.rollout import MockTauGymEnv, run_episode_rollout


class DummyPolicy:
    """Dummy policy that returns a tool call on turn 1 and final message on turn 2."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, history: list[dict[str, Any]]) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return "<think>Look up order.</think>\ncall:get_order(order_id='100')"
        return "<think>Done.</think>\nYour order #100 is resolved."


def test_mock_gym_env_reset_isolation() -> None:
    env = MockTauGymEnv(task_id="retail_001")
    obs1, info1 = env.reset()
    assert env.db_state == {"orders": {"100": "pending"}}

    # Mutate DB
    env.step("call:cancel_order(order_id='100')")
    assert env.db_state["orders"]["100"] == "cancelled"

    # Reset must restore pure state
    obs2, info2 = env.reset()
    assert env.db_state["orders"]["100"] == "pending"


def test_run_episode_rollout() -> None:
    env = MockTauGymEnv(task_id="retail_001")
    policy = DummyPolicy()

    trajectory = run_episode_rollout(env, policy, max_turns=5)

    assert trajectory["task_id"] == "retail_001"
    assert trajectory["num_turns"] == 2
    assert trajectory["terminated"] is True
    assert trajectory["reward"].reward == 1.0
    # Prior assistant reasoning stripped in accumulated history
    assert "<think>" not in trajectory["history"][2]["content"]
