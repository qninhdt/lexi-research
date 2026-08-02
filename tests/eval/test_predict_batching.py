"""Tests that batched generation returns what per-row generation returned.

Batching eval is a speed change, so the invariant under test is that it does not
move any measured number: the same completions, the same predictions, and the
same retry counts as decoding one row at a time.
"""

from __future__ import annotations

import json

import torch

from lexi_research.cli.config import load_config
from lexi_research.eval.predict import _left_pad, predict_rows
from lexi_research.format import BandConfig, default_config_path

ROWS = [
    {
        "target": "bright",
        "definition": "full of light; shining strongly",
        "pos": "adjective",
        "text": "The room is very bright today.",
        "correction": "The room is very bright today.",
        "meaning": 4,
        "feedback": "Natural sentence with the correct meaning.",
    },
    {
        "target": "speak",
        "definition": "to talk to someone about something",
        "pos": "verb",
        "text": "She speak very well.",
        "correction": "She [speak>speaks:agr] very well.",
        "meaning": 3,
        "feedback": "Subject-verb agreement needs attention.",
    },
    {
        "target": "quick",
        "definition": "moving fast",
        "pos": "adjective",
        "text": "He is quick.",
        "correction": "He is quick.",
        "meaning": 4,
        "feedback": "Correct and natural.",
    },
]


class _VaryingLengthTokenizer:
    """Emits a different prompt length per row, so padding actually happens.

    The row's own index is placed in the *last* prompt token. Left padding is
    what guarantees that token stays in the final column, so a right-padded
    batch would hand the model a pad token there and decode the wrong answer.
    """

    pad_token_id = 0

    def __init__(self) -> None:
        self.index = 0

    def apply_chat_template(self, _messages, **_kwargs):  # type: ignore[no-untyped-def]
        index = self.index
        self.index += 1
        # A different width per row, so the batch has real padding to place.
        width = 2 + index
        body = torch.arange(11, 11 + width - 1)
        return torch.cat((body, torch.tensor([500 + index]))).reshape(1, width)

    def decode(self, ids, **_kwargs):  # type: ignore[no-untyped-def]
        marker = int(ids[-1].item()) if len(ids) else 0
        index = marker - 500
        if 0 <= index < len(ROWS):
            row = ROWS[index]
            return json.dumps({key: row[key] for key in ("correction", "meaning", "feedback")})
        return "not json"


class _MarkerModel:
    """Echoes each row's identifying token, and records the batch widths it saw."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, *, input_ids, attention_mask, **_kwargs):  # type: ignore[no-untyped-def]
        self.batch_sizes.append(int(input_ids.shape[0]))
        # Left padding must put every real final prompt token in the last
        # column; the answer is keyed off it, so a right-padded batch would
        # read a pad token here and produce the wrong row's answer.
        assert torch.equal(attention_mask[:, -1], torch.ones(input_ids.shape[0], dtype=torch.long))
        return torch.cat((input_ids, input_ids[:, -1:]), dim=-1)


def _band_config() -> BandConfig:
    return BandConfig.from_json(default_config_path())


def test_left_pad_puts_real_tokens_last_and_masks_the_padding() -> None:
    prompts = [torch.tensor([[7, 8, 9]]), torch.tensor([[4, 5]])]

    ids, mask = _left_pad(prompts, pad_token_id=0, device=torch.device("cpu"))

    assert ids.tolist() == [[7, 8, 9], [0, 4, 5]]
    assert mask.tolist() == [[1, 1, 1], [0, 1, 1]]
    # Every row's last column is a real token, which is the property generate
    # continues from.
    assert mask[:, -1].tolist() == [1, 1]


def test_batched_prediction_matches_row_by_row_prediction() -> None:
    config = load_config(overrides=["train.thinking=off", "eval.max_new_tokens=4"])
    band_config = _band_config()

    batched_model = _MarkerModel()
    batched = predict_rows(
        config,
        ROWS,
        model=batched_model,
        tokenizer=_VaryingLengthTokenizer(),
        band_config=band_config,
        max_retries=0,
        batch_size=len(ROWS),
    )

    serial_model = _MarkerModel()
    serial = predict_rows(
        config,
        ROWS,
        model=serial_model,
        tokenizer=_VaryingLengthTokenizer(),
        band_config=band_config,
        max_retries=0,
        batch_size=1,
    )

    assert batched == serial
    assert batched_model.batch_sizes == [len(ROWS)]
    assert serial_model.batch_sizes == [1, 1, 1]


def test_a_row_answered_first_time_reports_no_retries() -> None:
    config = load_config(overrides=["train.thinking=off", "eval.max_new_tokens=4"])

    predictions = predict_rows(
        config,
        ROWS,
        model=_MarkerModel(),
        tokenizer=_VaryingLengthTokenizer(),
        band_config=_band_config(),
        max_retries=1,
        batch_size=2,
    )

    assert [p["retries"] for p in predictions] == [0, 0, 0]
    assert all(p["prediction"] is not None for p in predictions)


def test_only_unresolved_rows_are_regenerated_on_retry() -> None:
    config = load_config(overrides=["train.thinking=off", "eval.max_new_tokens=4"])

    class _AlwaysInvalidTokenizer(_VaryingLengthTokenizer):
        def decode(self, _ids, **_kwargs):  # type: ignore[no-untyped-def]
            return "no json object here"

    model = _MarkerModel()
    predictions = predict_rows(
        config,
        ROWS,
        model=model,
        tokenizer=_AlwaysInvalidTokenizer(),
        band_config=_band_config(),
        max_retries=1,
        batch_size=len(ROWS),
    )

    # Two attempts over all three rows, and the failure is reported rather than
    # hidden behind a None prediction with a zero retry count.
    assert model.batch_sizes == [len(ROWS), len(ROWS)]
    assert [p["retries"] for p in predictions] == [2, 2, 2]
    assert all(p["prediction"] is None for p in predictions)
