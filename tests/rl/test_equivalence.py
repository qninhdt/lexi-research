"""NRT with `seq_logp` must equal single-sample JEPO, exactly.

This is the most valuable test in the phase. The two rewards were written
independently — JEPO sums log-probabilities it is handed, NRT takes probabilities
and logs them itself — so a bug has to exist in both, identically, to pass. Every
other test here checks an implementation against an expectation; this one checks
two implementations against each other.

The equivalence is a claim in the design, not a coincidence of the code, so it is
asserted rather than assumed.
"""

from __future__ import annotations

import math

import pytest

from lexi_research.rl.jepo import jepo_reward, multi_sample_bound
from lexi_research.rl.nrt import nrt_reward, seq_logp

PROBABILITIES = [0.9, 0.5, 0.01, 0.75, 0.3, 0.999]


def test_nrt_seqlogp_equals_jepo() -> None:
    logprobs = [math.log(p) for p in PROBABILITIES]
    assert nrt_reward(PROBABILITIES, aggregation="seq_logp") == pytest.approx(
        jepo_reward(logprobs), abs=1e-12
    )


@pytest.mark.parametrize("length", [1, 2, 7, 64])
def test_the_equivalence_holds_at_every_length(length) -> None:
    probabilities = [1.0 / (index + 2) for index in range(length)]
    logprobs = [math.log(p) for p in probabilities]
    assert seq_logp(probabilities) == pytest.approx(jepo_reward(logprobs), abs=1e-12)


def test_the_equivalence_survives_a_near_zero_probability() -> None:
    """NRT floors before the log; JEPO is handed the floored value. They agree."""
    probabilities = [0.5, 0.0, 0.5]
    floored = [max(p, 1e-12) for p in probabilities]
    assert seq_logp(probabilities) == pytest.approx(
        jepo_reward([math.log(p) for p in floored]), abs=1e-9
    )


def test_geo_mean_is_not_the_same_reward() -> None:
    """If it were, A4 would have nothing to measure."""
    assert nrt_reward(PROBABILITIES, aggregation="geo_mean") != pytest.approx(
        nrt_reward(PROBABILITIES, aggregation="seq_logp")
    )


def test_the_multi_sample_bound_reduces_to_the_single_sample_reward() -> None:
    """With K=1 the bound is the reward; otherwise the flag would change the arm."""
    assert multi_sample_bound([-3.5]) == pytest.approx(-3.5)


def test_the_bound_lies_between_the_mean_and_the_maximum() -> None:
    rewards = [-4.0, -2.0, -1.0]
    bound = multi_sample_bound(rewards)
    assert sum(rewards) / len(rewards) <= bound <= max(rewards)
