"""Group advantages and the empty-reasoning baseline.

`R(empty)` is not decoration. It is the primary diagnostic for the two endogenous
tracks: if the gap between a sampled reasoning and an empty one collapses to
zero, reasoning has stopped contributing and the run is dead however healthy the
raw reward curve looks.
"""

from __future__ import annotations

import math

import pytest

from lexi_research.rl.base import (
    Group,
    RLError,
    Rollout,
    advantages,
    baseline_clip,
    combined_loss,
    kl_divergence,
    step_report,
    zero_advantage_share,
)


def test_group_normalisation() -> None:
    """Hand-computed: mean 2.0, population std sqrt(2) over [0,2,2,4]."""
    rewards = [0.0, 2.0, 2.0, 4.0]
    expected = [(r - 2.0) / math.sqrt(2.0) for r in rewards]
    assert advantages(rewards) == pytest.approx(expected)
    assert sum(advantages(rewards)) == pytest.approx(0.0)


def test_a_constant_group_contributes_nothing() -> None:
    """No rollout was better than another, so there is nothing to learn from it.

    Normalising anyway would divide by a near-zero std and turn float noise into
    a large gradient.
    """
    assert advantages([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]


def test_a_group_that_differs_only_by_rounding_is_also_constant() -> None:
    assert advantages([1.0, 1.0 + 1e-15]) == [0.0, 0.0]


def test_empty_baseline_clipping() -> None:
    """A reasoning worse than no reasoning bought no improvement, not a negative one."""
    assert baseline_clip([0.2, 0.6, 0.9], 0.6) == [0.0, 0.0, pytest.approx(0.30000000000000004)]


def test_clipping_leaves_a_pure_improvement_untouched() -> None:
    assert baseline_clip([1.0, 2.0], 0.0) == [1.0, 2.0]


def test_a_group_entirely_below_the_baseline_is_dead() -> None:
    """Every rollout was worse than empty: after clipping there is no signal."""
    clipped = baseline_clip([0.1, 0.2], 0.5)
    assert clipped == [0.0, 0.0]
    assert advantages(clipped) == [0.0, 0.0]


def test_zero_advantage_share_counts_dead_groups() -> None:
    groups = [Group(rewards=(1.0, 1.0)), Group(rewards=(0.0, 1.0))]
    assert zero_advantage_share(groups) == pytest.approx(0.5)


def test_kl_is_zero_when_the_policy_has_not_moved() -> None:
    logprobs = [-0.1, -2.0, -0.5]
    assert kl_divergence(logprobs, logprobs) == pytest.approx(0.0)


def test_kl_is_positive_when_the_policy_is_more_confident() -> None:
    assert kl_divergence([-0.1, -0.1], [-1.0, -1.0]) == pytest.approx(0.9)


def test_misaligned_kl_inputs_raise() -> None:
    with pytest.raises(RLError):
        kl_divergence([-0.1], [-0.1, -0.2])


def test_combined_loss_is_additive_in_lambda() -> None:
    """One interpretable knob to sweep, which is why the terms are not interleaved."""
    assert combined_loss(2.0, 4.0, lambda_rl=0.0) == pytest.approx(2.0)
    assert combined_loss(2.0, 4.0, lambda_rl=0.5) == pytest.approx(4.0)
    assert combined_loss(2.0, 4.0, lambda_rl=0.5, kl=1.0, kl_coefficient=0.1) == pytest.approx(4.1)


def test_step_report_carries_every_health_field() -> None:
    groups = [Group(rewards=(0.2, 0.8), empty_reward=0.3)]
    rollouts = [Rollout("because", 0.2, {}, 12), Rollout("since", 0.8, {}, 20)]
    report = step_report(groups, rollouts, kl=0.05).as_dict()
    assert set(report) == {
        "rl/reward_mean",
        "rl/reward_std",
        "rl/advantage_std",
        "rl/empty_reward",
        "rl/empty_gap",
        "rl/kl",
        "rl/reasoning_tokens",
        "rl/zero_advantage_share",
    }
    assert report["rl/reward_mean"] == pytest.approx(0.5)
    assert report["rl/empty_gap"] == pytest.approx(0.2)
    assert report["rl/reasoning_tokens"] == pytest.approx(16.0)


def test_the_empty_gap_is_what_says_a_run_is_dead() -> None:
    """Reasoning that buys nothing shows here even while reward looks healthy."""
    groups = [Group(rewards=(0.9, 0.9), empty_reward=0.9)]
    report = step_report(groups, [], kl=0.0)
    assert report.reward_mean == pytest.approx(0.9)
    assert report.empty_gap == pytest.approx(0.0)
    assert report.zero_advantage_share == pytest.approx(1.0)


def test_an_empty_group_raises() -> None:
    with pytest.raises(RLError):
        Group(rewards=())
