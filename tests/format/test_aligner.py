from __future__ import annotations

import pytest
from lexi_research.eval.harness import iter_rows
from lexi_research.format.aligner import (
    AlignedEdit,
    align_words,
    annotated_to_corrected,
    reconstruct_source,
)


def test_annotated_to_corrected_basic_cases():
    # Replace
    assert (
        annotated_to_corrected("He [speak>speaks:agr] English.")
        == "He speaks English."
    )
    # Multi-word replace
    assert (
        annotated_to_corrected("I [have seen>saw:tense] him in 1990.")
        == "I saw him in 1990."
    )
    # Deletion
    assert (
        annotated_to_corrected("I like [the>:art] music.")
        == "I like music."
    )
    # Insertion
    assert (
        annotated_to_corrected("I went [>to:prep] school.")
        == "I went to school."
    )
    # Clean sentence
    assert (
        annotated_to_corrected("He speaks English well.")
        == "He speaks English well."
    )
    # Null sentence
    assert annotated_to_corrected("null") == "null"
    assert annotated_to_corrected(None) == "null"


def test_reconstruct_source_basic_cases():
    assert (
        reconstruct_source("He [speak>speaks:agr] English.")
        == "He speak English."
    )
    assert (
        reconstruct_source("I [have seen>saw:tense] him in 1990.")
        == "I have seen him in 1990."
    )
    assert (
        reconstruct_source("I like [the>:art] music.")
        == "I like the music."
    )
    assert (
        reconstruct_source("I went [>to:prep] school.")
        == "I went school."
    )
    assert (
        reconstruct_source("He speaks English well.")
        == "He speaks English well."
    )
    assert reconstruct_source("null") == "null"
    assert reconstruct_source(None) == "null"


def test_align_words_basic():
    # Single replace
    edits = align_words("He speak English.", "He speaks English.")
    assert edits == [
        AlignedEdit(start=2, end=3, original="speak", replacement="speaks")
    ]

    # Multi-word replace
    edits = align_words("I have seen him in 1990.", "I saw him in 1990.")
    assert edits == [
        AlignedEdit(start=2, end=4, original="have seen", replacement="saw")
    ]

    # Deletion
    edits = align_words("I like the music.", "I like music.")
    assert edits == [
        AlignedEdit(start=3, end=4, original="the", replacement="")
    ]

    # Insertion
    edits = align_words("I went school.", "I went to school.")
    assert edits == [
        AlignedEdit(start=3, end=3, original="", replacement="to")
    ]

    # Clean sentence
    edits = align_words("He speaks English well.", "He speaks English well.")
    assert edits == []

    # Multiple edits
    edits = align_words(
        "He speak English very good.", "He speaks English very well."
    )
    assert edits == [
        AlignedEdit(start=2, end=3, original="speak", replacement="speaks"),
        AlignedEdit(start=5, end=6, original="good", replacement="well"),
    ]


def test_100_percent_source_reconstruction_on_dataset():
    """Verify 100.0000% exact match source reconstruction across the entire dataset."""
    train_rows = list(iter_rows("data/gec/train.parquet"))
    val_rows = list(iter_rows("data/gec/val.parquet"))
    all_rows = train_rows + val_rows

    assert len(all_rows) >= 19000

    mismatches = []
    for idx, r in enumerate(all_rows):
        raw = r["text"].strip()
        gold = r["correction"].strip()
        if gold.lower() == "null":
            continue
        recon = reconstruct_source(gold)
        if recon != raw:
            mismatches.append((idx, raw, gold, recon))

    assert (
        len(mismatches) == 0
    ), f"Found {len(mismatches)} mismatches in source reconstruction!"
