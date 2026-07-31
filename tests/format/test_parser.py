"""Parser tests: the three operations, the two properties, and adversarial input.

The parser is the one place untrusted model output meets our code, so the
malformed cases matter as much as the happy path: it must return a typed failure
for anything it cannot read, never raise.
"""

from __future__ import annotations

import pytest

from lexi_research.format import (
    Edit,
    ParseError,
    ParseOk,
    parse_correction,
    render,
    strip_markup,
)

# (correction, stripped text) pairs that must parse and round-trip.
VALID_CASES: list[tuple[str, str]] = [
    # A clean sentence passes through untouched — the zero-overhead case.
    ("The room has bright light.", "The room has bright light."),
    # Replace.
    ("He [speak>speaks:agr] clearly.", "He speak clearly."),
    # Delete.
    ("the [the>:art] very good idea", "the the very good idea"),
    # Insert, mid-sentence.
    ("went [>to the:art] store", "went store"),
    # Insert, sentence-initial.
    ("[>I:pron] went home", "went home"),
    # Insert, sentence-final.
    ("I went [>home:other]", "I went"),
    # Several edits of mixed kinds in one sentence.
    (
        "The room [have>has:agr] bright light in [>the:art] morning.",
        "The room have bright light in morning.",
    ),
    # Two adjacent replacements.
    ("She [speak>speaks:agr] very [eloquent>eloquently:form].", "She speak very eloquent."),
    # A literal bracket in learner text, escaped. Both brackets escape, so the
    # closing one cannot terminate an edit early.
    (r"I wrote \[sic\] there.", "I wrote [sic] there."),
    (r"I wrote \] alone.", "I wrote ] alone."),
    # An escaped bracket inside an edit field.
    (r"he said \[hi\] [loud>loudly:form]", "he said [hi] loud"),
    # `>` outside a bracket group is ordinary text.
    ("5 > 3 is [true>correct:word].", "5 > 3 is true."),
    # An empty correction is a (degenerate) clean sentence.
    ("", ""),
]

MALFORMED_CASES: list[tuple[str, str]] = [
    ("He [speak>speaks:agr clearly.", "malformed_edit"),  # unclosed
    ("He [speak>speaks:agreement] clearly.", "unknown_tag"),  # tag outside the set
    ("He [speak>speaks:AGR] clearly.", "malformed_edit"),  # uppercase fails the regex
    ("He [speak>speaks] clearly.", "malformed_edit"),  # no tag
    ("He [>:agr] clearly.", "empty_edit"),  # neither side
    ("He [[speak>speaks:agr]] clearly.", "malformed_edit"),  # nested brackets
    ("He [sp[eak>speaks:agr] clearly.", "malformed_edit"),  # bracket inside a field
]


@pytest.mark.parametrize(("correction", "text"), VALID_CASES)
def test_parses_and_strips_to_the_learner_text(correction: str, text: str) -> None:
    result = parse_correction(correction)
    assert isinstance(result, ParseOk)
    assert result.text == text
    assert strip_markup(correction) == text


@pytest.mark.parametrize(("correction", "text"), VALID_CASES)
def test_round_trips_through_render(correction: str, text: str) -> None:
    """parse → render reproduces the correction for every canonical input."""
    result = parse_correction(correction)
    assert isinstance(result, ParseOk)
    assert render(result.text, result.edits) == correction


@pytest.mark.parametrize(("correction", "text"), VALID_CASES)
def test_spans_locate_each_edit_in_the_stripped_text(correction: str, text: str) -> None:
    result = parse_correction(correction)
    assert isinstance(result, ParseOk)
    for edit in result.edits:
        start, end = edit.span
        assert result.text[start:end] == edit.original


@pytest.mark.parametrize(("correction", "code"), MALFORMED_CASES)
def test_malformed_input_returns_a_typed_error(correction: str, code: str) -> None:
    result = parse_correction(correction)
    assert isinstance(result, ParseError)
    assert result.code == code
    assert result.detail


@pytest.mark.parametrize(
    "correction",
    [
        "[",
        "]",
        "[]",
        "[>",
        "[a>b:]",
        "[a>b:tag",
        "\\",
        "[a>b:agr",
        "text [a>b:agr] more [",
        "[" * 50,
        "[a>b:agr]" * 200,
        "\x00 null byte [a>b:agr]",
        "múltiple ünicode [ü>u:sp] here",
    ],
)
def test_never_raises_on_adversarial_input(correction: str) -> None:
    """Whatever a model emits, the parser returns a value rather than raising."""
    result = parse_correction(correction)
    assert isinstance(result, ParseOk | ParseError)


def test_edit_fields_carry_the_operation() -> None:
    result = parse_correction("a [x>y:sp] b [>z:art] c [w>:punc] d")
    assert isinstance(result, ParseOk)
    replace, insert, delete = result.edits

    assert (replace.original, replace.replacement, replace.tag) == ("x", "y", "sp")
    assert not replace.is_insert and not replace.is_delete

    assert insert.is_insert and insert.original == "" and insert.replacement == "z"
    assert insert.span[0] == insert.span[1]  # zero-width

    assert delete.is_delete and delete.replacement == ""


def test_escaped_bracket_inside_an_edit_field() -> None:
    result = parse_correction(r"he said [\[hi\]>hi:punc] loudly")
    assert isinstance(result, ParseOk)
    assert result.text == "he said [hi] loudly"
    assert result.edits[0].original == "[hi]"
    assert render(result.text, result.edits) == r"he said [\[hi\]>hi:punc] loudly"


def test_strip_markup_raises_only_for_trusted_callers() -> None:
    with pytest.raises(ValueError, match="unknown_tag"):
        strip_markup("He [speak>speaks:nope] clearly.")


def test_render_rejects_a_span_that_does_not_hold_its_original() -> None:
    with pytest.raises(ValueError, match="does not hold"):
        render("He speaks", [Edit(original="walk", replacement="walks", tag="agr", span=(3, 7))])


def test_render_rejects_overlapping_spans() -> None:
    edits = [
        Edit(original="ab", replacement="x", tag="sp", span=(0, 2)),
        Edit(original="bc", replacement="y", tag="sp", span=(1, 3)),
    ]
    with pytest.raises(ValueError, match="out of order"):
        render("abcd", edits)


def test_render_rejects_a_span_past_the_end_of_the_text() -> None:
    with pytest.raises(ValueError, match="out of range"):
        render("ab", [Edit(original="", replacement="x", tag="art", span=(5, 5))])


def test_a_literal_bracket_in_learner_text_renders_escaped() -> None:
    r"""`render` must escape brackets in the plain-text segments too.

    Emitting them bare would produce a string that no longer parses back, which
    the round-trip property would only catch for inputs that already contain
    one — so it is asserted directly here.
    """
    edits = [Edit(original="loud", replacement="loudly", tag="form", span=(14, 18))]
    rendered = render("he said [sic] loud", edits)
    assert rendered == r"he said \[sic\] [loud>loudly:form]"
    assert strip_markup(rendered) == "he said [sic] loud"


def test_a_literal_bracket_in_an_edit_field_renders_escaped() -> None:
    """A bare `]` inside a field would close the markup early."""
    edits = [Edit(original="[x]", replacement="]y[", tag="punc", span=(0, 3))]
    rendered = render("[x] rest", edits)
    assert strip_markup(rendered) == "[x] rest"

    reparsed = parse_correction(rendered)
    assert isinstance(reparsed, ParseOk)
    assert reparsed.edits[0].original == "[x]"
    assert reparsed.edits[0].replacement == "]y["
