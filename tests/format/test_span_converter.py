from __future__ import annotations

import pytest
from lexi_research.eval.harness import iter_rows
from lexi_research.format.span_converter import (
    markup_to_spans,
    parse_span_output,
    render_spans_to_markup,
    validate_span_edits,
)
from lexi_research.format.units import SpanEdit, format_numbered_input, lex_units


def test_lex_units_basic():
    units = lex_units("He speak English very good.")
    assert [u.token for u in units] == ["He", "speak", "English", "very", "good", "."]
    assert [u.index for u in units] == [1, 2, 3, 4, 5, 6]
    assert format_numbered_input("He speak English.") == "1 He\n2 speak\n3 English\n4 ."


def test_lex_units_contractions_and_unicode():
    units = lex_units("I couldn't visit café Noir in 1990.")
    assert [u.token for u in units] == ["I", "could", "n't", "visit", "café", "Noir", "in", "1990", "."]


def test_single_word_replace():
    raw = "He speak English."
    gold = "He [speak>speaks:agr] English."
    spans = markup_to_spans(raw, gold)
    assert spans == "2 3 agr speaks"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == gold


def test_multi_word_replace():
    raw = "I have seen him in 1990."
    gold = "I [have seen>saw:tense] him in 1990."
    spans = markup_to_spans(raw, gold)
    assert spans == "2 4 tense saw"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == gold


def test_deletion():
    raw = "I like the music."
    gold = "I like [the>:art] music."
    spans = markup_to_spans(raw, gold)
    assert spans == "3 4 art"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == gold


def test_insertion():
    raw = "I went school."
    gold = "I went [>to:prep] school."
    spans = markup_to_spans(raw, gold)
    assert spans == "3 3 prep to"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == gold


def test_clean_sentence():
    raw = "He speaks English well."
    gold = "He speaks English well."
    spans = markup_to_spans(raw, gold)
    assert spans == "OK"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == raw


def test_null_sentence():
    raw = "asdf qwer zxcv"
    gold = "null"
    spans = markup_to_spans(raw, gold)
    assert spans == "NULL"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == "null"


def test_multiple_edits():
    raw = "He speak English very good."
    gold = "He [speak>speaks:agr] English very [good>well:word]."
    spans = markup_to_spans(raw, gold)
    assert spans == "2 3 agr speaks\n5 6 word well"

    rendered = render_spans_to_markup(raw, spans)
    assert rendered == gold


def test_validate_span_edits():
    units = lex_units("He speak English.")
    valid, msg = validate_span_edits("2 3 agr speaks", len(units))
    assert valid is True

    valid, msg = validate_span_edits("OK", len(units))
    assert valid is True

    # Out of bounds
    valid, msg = validate_span_edits("10 12 agr speaks", len(units))
    assert valid is False

    # Invalid tag
    valid, msg = validate_span_edits("2 3 notatag speaks", len(units))
    assert valid is False


def test_dataset_100_percent_roundtrip():
    """Verify 100.0000% exact roundtrip match across all dataset samples."""
    train_rows = list(iter_rows("data/gec/train.parquet"))
    val_rows = list(iter_rows("data/gec/val.parquet"))
    all_rows = train_rows + val_rows

    assert len(all_rows) >= 19000

    mismatches = []
    for idx, r in enumerate(all_rows):
        raw = r["text"]
        gold = r["correction"]
        spans = markup_to_spans(raw, gold)
        rendered = render_spans_to_markup(raw, spans)
        if rendered.strip() != gold.strip():
            mismatches.append((idx, raw, gold, rendered))

    assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches in roundtrip conversion!"
