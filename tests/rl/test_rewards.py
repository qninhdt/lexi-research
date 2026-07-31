"""The verifiable reward.

Exogenous by construction: nothing here depends on the policy, which is why GRPO
runs first. The reward is computed by the same `format` primitives the eval
harness scores with — a reward that measured something slightly different would
train the model toward a number nobody reports.
"""

from __future__ import annotations

import pytest

from lexi_research.rl.rewards import (
    RewardError,
    RewardWeights,
    normalised_reward,
    parse_answer,
    verifiable_reward,
)

ROW = {
    "text": "She speak very well today.",
    "correction": "She [speak>spoke:tense] very well today.",
    "meaning": 4,
    "feedback": "Good meaning, only the tense needs fixing.",
}
PERFECT = {
    "correction": "She [speak>spoke:tense] very well today.",
    "meaning": 4,
    "feedback": "Good meaning, only the tense needs fixing.",
}


def test_perfect_prediction_scores_one(config) -> None:
    parts = verifiable_reward(PERFECT, ROW, config)
    assert parts.edit_f1 == pytest.approx(1.0)
    assert parts.meaning == pytest.approx(1.0)
    assert parts.format_valid == pytest.approx(1.0)
    assert parts.strip_mismatch == pytest.approx(0.0)
    assert normalised_reward(parts) == pytest.approx(1.0)


def test_monotone_degradation(config) -> None:
    """Reward falls as the answer is progressively corrupted, never rises."""
    ladder = [
        PERFECT,
        {**PERFECT, "meaning": 3},
        {**PERFECT, "meaning": 2},
        {**PERFECT, "meaning": 2, "correction": "She speak very well today."},
        {**PERFECT, "meaning": 0, "correction": "She speak very well today."},
    ]
    scores = [verifiable_reward(step, ROW, config).total for step in ladder]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_a_wrong_tag_costs_less_than_a_wrong_span(config) -> None:
    wrong_tag = {**PERFECT, "correction": "She [speak>spoke:sp] very well today."}
    wrong_span = {**PERFECT, "correction": "She speak [very>quite:word] well today."}
    assert (
        verifiable_reward(wrong_span, ROW, config).total
        <= verifiable_reward(wrong_tag, ROW, config).total
    )


def test_a_correction_that_rewrites_unmarked_text_is_penalised(config) -> None:
    """The one failure invisible downstream: every span metric still agrees."""
    drifted = {**PERFECT, "correction": "She [speak>spoke:tense] very well indeed."}
    parts = verifiable_reward(drifted, ROW, config)
    assert parts.strip_mismatch == 1.0
    assert parts.total < 0.0


def test_an_unparseable_answer_scores_zero(config) -> None:
    assert verifiable_reward(None, ROW, config).total == 0.0
    assert parse_answer("not json at all") is None
    assert parse_answer('["a list"]') is None


def test_a_meaning_outside_the_band_range_scores_zero(config) -> None:
    assert verifiable_reward({**PERFECT, "meaning": 9}, ROW, config).meaning == 0.0


def test_a_boolean_meaning_is_not_band_one(config) -> None:
    """`bool` is an `int` subclass, and `True` must not pass as band 1."""
    assert verifiable_reward({**PERFECT, "meaning": True}, ROW, config).meaning == 0.0


def test_the_meaning_term_falls_linearly(config) -> None:
    scores = [verifiable_reward({**PERFECT, "meaning": b}, ROW, config).meaning for b in range(5)]
    assert scores == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_feedback_is_excluded_from_the_default_scope(config) -> None:
    """A3's central claim: feedback is unverifiable, so rewarding it teaches the
    model to chase the teacher's phrasing rather than its grading."""
    different = {**PERFECT, "feedback": "Completely unrelated wording here."}
    assert (
        verifiable_reward(different, ROW, config).total
        == verifiable_reward(PERFECT, ROW, config).total
    )


def test_the_full_answer_scope_does_reward_feedback(config) -> None:
    """A3's other arm — otherwise there would be nothing to compare."""
    different = {**PERFECT, "feedback": "Completely unrelated wording here."}
    matched = verifiable_reward(PERFECT, ROW, config, scope="full_answer").total
    unmatched = verifiable_reward(different, ROW, config, scope="full_answer").total
    assert matched > unmatched


def test_an_unknown_scope_raises(config) -> None:
    with pytest.raises(RewardError, match="scope"):
        verifiable_reward(PERFECT, ROW, config, scope="everything")


def test_weights_come_from_config() -> None:
    weights = RewardWeights.from_config({"edit_f1": 1.0, "meaning": 0.0})
    assert weights.edit_f1 == 1.0
    assert weights.meaning == 0.0
    with pytest.raises(RewardError, match="unknown"):
        RewardWeights.from_config({"edit_f2": 1.0})


def test_the_reward_uses_the_same_edit_f1_the_harness_reports() -> None:
    """A reward that reimplemented the metric could reward what eval never sees."""
    import inspect

    from lexi_research.rl import rewards

    source = inspect.getsource(rewards)
    assert "from lexi_research.data.pilot_gate import edit_f1" in source
    assert "def edit_f1" not in source


def test_a_drifted_correction_is_never_worth_emitting(config) -> None:
    """The penalty exceeds the whole positive total, so no combination of good
    edit-F1 and good meaning makes rewriting unmarked text profitable."""
    weights = RewardWeights()
    assert weights.strip_mismatch > weights.positive_total - weights.format_valid
    drifted = {**PERFECT, "correction": "She [speak>spoke:tense] very well indeed."}
    assert verifiable_reward(drifted, ROW, config).total < 0.0
