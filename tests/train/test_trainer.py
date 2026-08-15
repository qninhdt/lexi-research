"""Trainer plumbing that does not need a model: loading rows, and the drop ceiling.

Dropping over-long rows is length-biased — short learner sentences and short
feedback survive — so a run that silently drops most of its data trains on a
different distribution than the one it reports.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.train.trainer import TrainerSetupError, build_examples, load_rows

from .test_collate import ROW, StubTokenizer


def test_load_rows_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(ROW) + "\n\n" + json.dumps(ROW) + "\n", encoding="utf-8")
    assert load_rows(path) == [ROW, ROW]


def test_load_rows_reads_parquet(tmp_path) -> None:
    """The format the real dataset arrives in."""
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "rows.parquet"
    pq.write_table(pyarrow.table({key: [value] for key, value in ROW.items()}), path)
    assert load_rows(path) == [ROW]


def test_load_rows_names_a_missing_file() -> None:
    with pytest.raises(TrainerSetupError, match="does not exist"):
        load_rows("no/such/file.jsonl")


def _build(rows, **kwargs):
    return build_examples(
        StubTokenizer(),
        rows,
        max_seq_len=kwargs.pop("max_seq_len", 100_000),
        thinking="on",
        completion_only=True,
        **kwargs,
    )


def test_every_row_over_the_limit_raises() -> None:
    with pytest.raises(TrainerSetupError, match="every one of"):
        _build([ROW] * 3, max_seq_len=4)


def _limit_admitting_only_short(short: dict, long: dict) -> int:
    """A `max_seq_len` that fits `short` and not `long`, whatever the prompt costs.

    Derived rather than hardcoded: a literal here silently becomes wrong when the
    grader rubric changes length, and it then fails as "every row is too long"
    instead of testing the drop ceiling it was written for.
    """
    fits, _ = _build([short])
    too_long, _ = _build([long])
    short_len = len(fits[0].input_ids)
    long_len = len(too_long[0].input_ids)
    assert short_len < long_len, "the fixture rows must differ in length"
    return (short_len + long_len) // 2


def test_dropping_over_long_rows_is_reported_not_fatal() -> None:
    short = dict(ROW, feedback="Fine.")
    long = dict(ROW, text=ROW["text"] + " word" * 400)
    limit = _limit_admitting_only_short(short, long)

    examples, dropped = _build([short] + [long] * 4, max_seq_len=limit)
    assert dropped == 4
    assert len(examples) == 1


def test_supervised_tokens_are_a_small_share_of_the_sequence() -> None:
    """The point of the completion-only mask: the rubric is not the lesson."""
    examples, _ = _build([ROW])
    example = examples[0]
    assert 0 < example.supervised_tokens < len(example.input_ids) * 0.2
