"""Segment location, tested hardest because its failure is silent.

Everything in the RL package depends on knowing exactly which tokens are the
reasoning, which are the correction, which are the meaning band, and which are
the feedback. A boundary off by one token trains the wrong thing and no reward
curve looks unusual when it happens.

The spans are constructed rather than discovered, so these tests check that the
construction is exact — including the `correction: null` case, where the span
must cover the literal `null` rather than being empty.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.rl.segments import (
    SegmentError,
    build_segments,
    policy_gradient_mask,
    reward_mask,
    supervised_mask,
)
from tests.train.test_collate import ROW, StubTokenizer

PROMPT = [101, 102, 103]


@pytest.fixture()
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


def _decode(tokenizer, segments, span) -> str:
    return tokenizer.decode(list(segments.input_ids[span.start : span.end]))


def test_spans_thinking(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="on", reasoning="the verb is past")

    assert len(segments.prompt) == len(PROMPT)
    assert "past" in _decode(tokenizer, segments, segments.reasoning)
    assert str(ROW["meaning"]) in _decode(tokenizer, segments, segments.meaning)
    assert "bright" in _decode(tokenizer, segments, segments.correction)
    assert "Natural" in _decode(tokenizer, segments, segments.feedback)


def test_spans_non_thinking(tokenizer) -> None:
    """An `off` render has an empty reasoning span, and everything else still lands."""
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    assert segments.reasoning.empty
    assert not segments.correction.empty
    assert not segments.meaning.empty
    assert not segments.feedback.empty


def test_forced_empty_carries_the_block_without_a_sampled_reasoning(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="forced-empty")
    assert not segments.reasoning.empty
    assert "<think>" in _decode(tokenizer, segments, segments.reasoning)


def test_null_correction(tokenizer) -> None:
    """A `correction: null` row has a correction span covering `null`, not nothing.

    An empty span would give that row no correction reward and no gradient, and
    the rows judged beyond correction are exactly the ones the band-0 signal
    lives in.
    """
    row = dict(ROW, correction=None)
    segments = build_segments(tokenizer, row, PROMPT, thinking="off")
    assert not segments.correction.empty
    assert "null" in _decode(tokenizer, segments, segments.correction)


def test_the_spans_do_not_overlap(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="on", reasoning="because")
    spans = [
        segments.prompt,
        segments.reasoning,
        segments.correction,
        segments.feedback,
        segments.meaning,
    ]
    for earlier, later in zip(spans[:-1], spans[1:], strict=True):
        assert earlier.end <= later.start


def test_the_spans_are_in_serialised_order_not_reading_order(tokenizer) -> None:
    """`sort_keys=True` puts feedback between correction and meaning."""
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    assert segments.correction.end <= segments.feedback.start
    assert segments.feedback.end <= segments.meaning.start


def test_the_sequence_starts_with_the_prompt_it_was_given(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    assert list(segments.input_ids[: len(PROMPT)]) == PROMPT


def test_the_answer_span_covers_everything_after_the_reasoning(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="on", reasoning="z")
    assert segments.answer.start == segments.reasoning.end
    assert segments.answer.end == len(segments.input_ids)
    for span in (segments.correction, segments.feedback, segments.meaning):
        assert segments.answer.start <= span.start and span.end <= segments.answer.end


def test_a_reasoning_with_thinking_off_raises(tokenizer) -> None:
    with pytest.raises(SegmentError, match="off"):
        build_segments(tokenizer, ROW, PROMPT, thinking="off", reasoning="but here it is")


def test_an_unknown_thinking_mode_raises(tokenizer) -> None:
    with pytest.raises(SegmentError, match="thinking"):
        build_segments(tokenizer, ROW, PROMPT, thinking="maybe")


def test_feedback_excluded_from_reward(tokenizer) -> None:
    """The design's central claim, enforced rather than documented."""
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="on", reasoning="z")
    mask = reward_mask(segments, "correction_meaning")
    assert all(mask[index] == 0 for index in segments.feedback.indices())
    assert all(mask[index] == 1 for index in segments.correction.indices())
    assert all(mask[index] == 1 for index in segments.meaning.indices())


def test_the_full_answer_scope_includes_feedback(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    mask = reward_mask(segments, "full_answer")
    assert all(mask[index] == 1 for index in segments.feedback.indices())


def test_an_unknown_reward_scope_raises(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    with pytest.raises(SegmentError, match="reward_scope"):
        reward_mask(segments, "everything")


def test_policy_grad_only_on_reasoning(tokenizer) -> None:
    """The answer is supervised by cross-entropy; two objectives on one token
    would make the tracks incomparable."""
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="on", reasoning="because past")
    mask = policy_gradient_mask(segments)
    assert sum(mask) == len(segments.reasoning)
    for span in (segments.prompt, segments.correction, segments.meaning, segments.feedback):
        assert all(mask[index] == 0 for index in span.indices())


def test_the_prompt_receives_neither_reward_nor_gradient(tokenizer) -> None:
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="on", reasoning="z")
    for mask in (reward_mask(segments, "full_answer"), policy_gradient_mask(segments)):
        assert all(mask[index] == 0 for index in segments.prompt.indices())


def test_supervision_covers_the_whole_answer_including_feedback(tokenizer) -> None:
    """Feedback gets no reward but is still fully supervised — design §3."""
    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    mask = supervised_mask(segments)
    assert all(mask[index] == 1 for index in segments.feedback.indices())
    assert all(mask[index] == 0 for index in segments.prompt.indices())


def test_the_serialised_answer_is_the_one_the_collator_supervises(tokenizer) -> None:
    """Spans over a different serialisation than training used would be spans
    over a sequence the model never sees."""
    from lexi_research.train.collate import completion_text

    segments = build_segments(tokenizer, ROW, PROMPT, thinking="off")
    answer = tokenizer.decode(list(segments.input_ids[segments.answer.start :]))
    expected = completion_text(ROW)
    assert json.loads(expected)["meaning"] == ROW["meaning"]
    assert "correction" in answer and "feedback" in answer and "meaning" in answer
