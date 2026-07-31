"""Validator tests: one per check, each failing in isolation.

Every test starts from a payload that passes all six checks and breaks exactly
one thing, so a failure names the check that regressed rather than "validation".
"""

from __future__ import annotations

from typing import Any

import pytest

from lexi_research.format import (
    BandConfig,
    ValidationError,
    ValidationOk,
    validate_output,
)

INPUT_TEXT = "The room have bright light in morning."
CORRECTION = "The room [have>has:agr] bright light in [>the:art] morning."


def payload(**overrides: Any) -> dict[str, Any]:
    """A payload that passes all six checks, with fields overridden per test."""
    base: dict[str, Any] = {
        "correction": CORRECTION,
        "meaning": 4,
        "feedback": "Good use of 'bright', but check subject-verb agreement.",
    }
    base.update(overrides)
    return base


def test_accepts_a_well_formed_payload(config: BandConfig) -> None:
    result = validate_output(payload(), INPUT_TEXT, config)
    assert isinstance(result, ValidationOk)
    assert result.meaning == 4
    assert result.edits is not None
    assert [edit.tag for edit in result.edits] == ["agr", "art"]
    assert not result.unparseable
    # Bands come attached, so callers never re-derive them.
    assert result.bands.naturalness == 4


def test_accepts_a_clean_sentence_with_no_edits(config: BandConfig) -> None:
    text = "The room has bright light in the morning."
    result = validate_output(payload(correction=text), text, config)
    assert isinstance(result, ValidationOk)
    assert result.edits == []
    assert result.bands.grammar == 4


def test_accepts_a_null_correction_and_floors_grammar(config: BandConfig) -> None:
    """`null` is the honest answer for an unparseable sentence, not a failure."""
    result = validate_output(payload(correction=None), INPUT_TEXT, config)
    assert isinstance(result, ValidationOk)
    assert result.unparseable
    assert result.bands.grammar == 0


@pytest.mark.parametrize("field", ["correction", "meaning", "feedback"])
def test_rejects_a_missing_field(config: BandConfig, field: str) -> None:
    incomplete = payload()
    del incomplete[field]
    result = validate_output(incomplete, INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "missing_field"


def test_check_1_rejects_a_correction_that_does_not_parse(config: BandConfig) -> None:
    broken = CORRECTION.replace("[have>has:agr]", "[have>has:agr")
    result = validate_output(payload(correction=broken), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "malformed_edit"


def test_check_2_rejects_a_tag_outside_the_taxonomy(config: BandConfig) -> None:
    result = validate_output(
        payload(correction=CORRECTION.replace(":agr]", ":agreement]")),
        INPUT_TEXT,
        config,
    )
    assert isinstance(result, ValidationError)
    assert result.code == "unknown_tag"


def test_check_3_rejects_text_the_model_altered_without_marking_it(
    config: BandConfig,
) -> None:
    """The check that cannot be dropped: whole-sentence re-emit hides silent rewrites."""
    smuggled = CORRECTION.replace("bright light", "brilliant light")
    result = validate_output(payload(correction=smuggled), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "text_altered"


def test_check_3_catches_a_dropped_word(config: BandConfig) -> None:
    truncated = "The room [have>has:agr] bright light in morning."
    result = validate_output(payload(correction=truncated), INPUT_TEXT + " Extra.", config)
    assert isinstance(result, ValidationError)
    assert result.code == "text_altered"


@pytest.mark.parametrize("meaning", [-1, 5, 42])
def test_check_4_rejects_a_meaning_outside_the_band_range(config: BandConfig, meaning: int) -> None:
    result = validate_output(payload(meaning=meaning), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "meaning_range"


@pytest.mark.parametrize("meaning", ["4", 4.0, None, True])
def test_check_4_rejects_a_non_integer_meaning(config: BandConfig, meaning: Any) -> None:
    """`True` is an int subclass and must not slip through as band 1."""
    result = validate_output(payload(meaning=meaning), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "bad_type"


def test_check_5_rejects_an_empty_edit(config: BandConfig) -> None:
    result = validate_output(
        payload(correction=CORRECTION.replace("[have>has:agr]", "[>:agr]have")),
        INPUT_TEXT,
        config,
    )
    assert isinstance(result, ValidationError)
    assert result.code == "empty_edit"


@pytest.mark.parametrize("feedback", ["", "   ", "\n\t"])
def test_check_6_rejects_empty_feedback(config: BandConfig, feedback: str) -> None:
    result = validate_output(payload(feedback=feedback), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "feedback_empty"


def test_check_6_rejects_multi_sentence_feedback(config: BandConfig) -> None:
    result = validate_output(
        payload(feedback="Good sense. But fix the verb."),
        INPUT_TEXT,
        config,
    )
    assert isinstance(result, ValidationError)
    assert result.code == "feedback_multi_sentence"


def test_check_6_accepts_one_sentence_with_a_terminator(config: BandConfig) -> None:
    result = validate_output(payload(feedback="Fix the verb agreement."), INPUT_TEXT, config)
    assert isinstance(result, ValidationOk)
    assert result.feedback == "Fix the verb agreement."


def test_check_6_accepts_an_abbreviation_inside_the_sentence(config: BandConfig) -> None:
    """An abbreviation's period is followed by a lowercase word, not a new sentence."""
    result = validate_output(
        payload(feedback="Use e.g. 'bright light' instead."), INPUT_TEXT, config
    )
    assert isinstance(result, ValidationOk)


@pytest.mark.parametrize("correction", [42, [], {}, True])
def test_rejects_a_correction_that_is_neither_string_nor_null(
    config: BandConfig, correction: Any
) -> None:
    result = validate_output(payload(correction=correction), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "bad_type"


def test_rejects_non_string_feedback(config: BandConfig) -> None:
    result = validate_output(payload(feedback=["fix the verb"]), INPUT_TEXT, config)
    assert isinstance(result, ValidationError)
    assert result.code == "bad_type"


def test_checks_run_in_order_so_the_first_failure_is_reported(config: BandConfig) -> None:
    """A payload broken in several ways reports the earliest check, not the last."""
    result = validate_output(
        payload(correction="[have>has:nope] wrong text", meaning=9, feedback=""),
        INPUT_TEXT,
        config,
    )
    assert isinstance(result, ValidationError)
    assert result.code == "unknown_tag"
