"""Tests for official outcome reward calculation and diagnostic metrics."""

from tau_research.tau.reward import TauReward, calculate_outcome_reward


def test_outcome_reward_all_success() -> None:
    reward_obj = calculate_outcome_reward(db_success=True, communicate_success=True)
    assert reward_obj.reward == 1.0
    assert reward_obj.is_success is True
    assert reward_obj.db_reward == 1.0
    assert reward_obj.communicate_reward == 1.0


def test_outcome_reward_db_fail() -> None:
    reward_obj = calculate_outcome_reward(db_success=False, communicate_success=True)
    assert reward_obj.reward == 0.0
    assert reward_obj.is_success is False
    assert reward_obj.db_reward == 0.0
    assert reward_obj.communicate_reward == 1.0


def test_outcome_reward_communicate_fail() -> None:
    reward_obj = calculate_outcome_reward(db_success=True, communicate_success=False)
    assert reward_obj.reward == 0.0
    assert reward_obj.is_success is False


def test_tau_reward_diagnostics() -> None:
    reward_obj = TauReward(
        reward=1.0,
        db_reward=1.0,
        communicate_reward=1.0,
        partial_action_reward=0.75,
        is_success=True,
    )
    d = reward_obj.to_dict()
    assert d["reward"] == 1.0
    assert d["partial_action_reward"] == 0.75
