"""The four-check correction validator, and why it is not the six-check one."""

from __future__ import annotations

import pytest

from lexi_research.format.validate_correction import (
    CorrectionError,
    CorrectionOk,
    validate_correction,
)


def ok(correction: str | None, text: str) -> CorrectionOk:
    result = validate_correction(correction, text)
    assert isinstance(result, CorrectionOk), result
    return result


def failure(correction: object, text: str) -> CorrectionError:
    result = validate_correction(correction, text)
    assert isinstance(result, CorrectionError), result
    return result


class TestAccepts:
    def test_a_clean_sentence_re_emitted_verbatim(self) -> None:
        assert ok("This is fine.", "This is fine.").edits == []

    def test_a_marked_edit(self) -> None:
        result = ok("He [speak>speaks:agr] well.", "He speak well.")
        assert result.edits is not None
        assert [edit.tag for edit in result.edits] == ["agr"]

    def test_null_means_unrecoverable(self) -> None:
        """No band is floored here — stage A has none. The caller decides."""
        result = ok(None, "asdf qwer zxcv")
        assert result.unparseable

    def test_an_insert_and_a_delete(self) -> None:
        assert ok("I went [>to:prep] school.", "I went school.").edits
        assert ok("I like [the>:art] music.", "I like the music.").edits


class TestRejects:
    def test_rewritten_untouched_text(self) -> None:
        """The failure nothing downstream can see: prose improved without marking."""
        assert failure("He [speak>speaks:agr] very well.", "He speak well.").code == (
            "strip_mismatch"
        )

    def test_a_tag_outside_the_taxonomy(self) -> None:
        assert failure("He [speak>speaks:oops] well.", "He speak well.").code == "unknown_tag"

    def test_malformed_markup(self) -> None:
        assert failure("He [speak>speaks well.", "He [speak>speaks well.").code == (
            "malformed_edit"
        )

    def test_a_non_string_correction(self) -> None:
        assert failure(42, "He speak well.").code == "correction_type"


def test_it_does_not_require_meaning_or_feedback() -> None:
    """The reason this module exists: a learner corpus supplies neither.

    `validate_output` would reject every stage-A row, and satisfying it would mean
    inventing a band and a feedback line for a sentence with no target word.
    """
    from lexi_research.format import ValidationError, validate_output

    payload = {"correction": "He [speak>speaks:agr] well."}
    assert isinstance(validate_output(payload, "He speak well.", _config()), ValidationError)
    assert isinstance(validate_correction(payload["correction"], "He speak well."), CorrectionOk)


def _config():
    from lexi_research.format import BandConfig, default_config_path

    return BandConfig.from_json(default_config_path())


@pytest.mark.parametrize(
    "correction",
    [
        "He [speak>speaks:agr] well.",
        "I went [>to:prep] school.",
        "I like [the>:art] music.",
        "Nothing to fix here.",
    ],
)
def test_accepted_corrections_round_trip(correction: str) -> None:
    from lexi_research.format import strip_markup

    assert isinstance(validate_correction(correction, strip_markup(correction)), CorrectionOk)
