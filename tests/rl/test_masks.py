"""The shared mask — the property that makes the three tracks comparable.

Design §3 assigns each segment to exactly one objective: cross-entropy over the
whole answer, reward over correction and meaning only, policy gradient over the
reasoning only. If any track deviated, a difference between two tracks would no
longer be attributable to its reward definition, which is the entire point of
running them against one another.
"""

from __future__ import annotations

import pytest

from lexi_research.rl.base import ALGORITHMS
from lexi_research.rl.segments import (
    build_segments,
    policy_gradient_mask,
    reward_mask,
    supervised_mask,
)
from tests.train.test_collate import ROW, StubTokenizer

PROMPT = [11, 12, 13]


@pytest.fixture()
def segments():
    return build_segments(
        StubTokenizer(), ROW, PROMPT, thinking="on", reasoning="the verb needs the past"
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_feedback_excluded_from_reward(algorithm, segments) -> None:
    """In every track. Feedback is voice and register: unverifiable, and
    rewarding it teaches the model to chase the teacher's phrasing."""
    mask = reward_mask(segments, "correction_meaning")
    assert all(mask[index] == 0 for index in segments.feedback.indices())
    assert algorithm in ALGORITHMS


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_policy_grad_only_on_reasoning(algorithm, segments) -> None:
    mask = policy_gradient_mask(segments)
    assert sum(mask) == len(segments.reasoning)
    assert all(mask[index] == 0 for index in segments.answer.indices())
    assert algorithm in ALGORITHMS


def test_the_three_objectives_never_share_a_token(segments) -> None:
    """Two objectives on one position is how a comparison stops being one."""
    reward = reward_mask(segments, "correction_meaning")
    gradient = policy_gradient_mask(segments)
    for a, b in zip(reward, gradient, strict=True):
        assert not (a and b)


def test_supervision_and_reward_overlap_by_design(segments) -> None:
    """Correction and meaning are both supervised and rewarded — that is §3."""
    supervised = supervised_mask(segments)
    reward = reward_mask(segments, "correction_meaning")
    assert all(supervised[index] for index, flag in enumerate(reward) if flag)


def test_the_prompt_is_in_no_mask(segments) -> None:
    for mask in (
        supervised_mask(segments),
        reward_mask(segments, "full_answer"),
        policy_gradient_mask(segments),
    ):
        assert all(mask[index] == 0 for index in segments.prompt.indices())


def test_every_answer_token_is_supervised(segments) -> None:
    supervised = supervised_mask(segments)
    covered = sum(supervised)
    assert covered == len(segments.correction) + len(segments.meaning) + len(segments.feedback)


def test_switching_the_reward_scope_changes_only_the_feedback_span(segments) -> None:
    """A3 is one axis: the scope must not move the other two objectives."""
    narrow = reward_mask(segments, "correction_meaning")
    wide = reward_mask(segments, "full_answer")
    differing = [index for index, (a, b) in enumerate(zip(narrow, wide, strict=True)) if a != b]
    assert differing == list(segments.feedback.indices())
