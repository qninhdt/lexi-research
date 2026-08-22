"""Tests for GRPOTrainer custom rollout_func and zero-variance detection."""

from tau_research.training.train_grpo import (
    check_zero_variance_reward_batch,
    format_rollout_batch_for_grpo,
)


def test_check_zero_variance_reward_batch() -> None:
    # All zero -> zero variance
    assert check_zero_variance_reward_batch([0.0, 0.0, 0.0, 0.0]) is True
    # All one -> zero variance
    assert check_zero_variance_reward_batch([1.0, 1.0, 1.0, 1.0]) is True
    # Mixed -> non-zero variance (informative signal)
    assert check_zero_variance_reward_batch([0.0, 1.0, 0.0, 1.0]) is False


def test_format_rollout_batch_for_grpo() -> None:
    prompt_ids = [[1, 2, 3], [1, 2, 3]]
    completion_ids = [[4, 5], [6, 7]]
    rewards = [1.0, 0.0]

    batch = format_rollout_batch_for_grpo(prompt_ids, completion_ids, rewards)
    assert "prompt_ids" in batch
    assert "completion_ids" in batch
    assert "rewards" in batch
    assert batch["rewards"] == [1.0, 0.0]
