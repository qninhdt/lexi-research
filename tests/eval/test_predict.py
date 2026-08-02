"""Inference adapter tests for tokenizer return shapes."""

from __future__ import annotations

import json

import torch

from lexi_research.cli.config import load_config
from lexi_research.eval.predict import predict_rows
from lexi_research.format import BandConfig, default_config_path

ROW = {
    "target": "bright",
    "definition": "full of light; shining strongly",
    "pos": "adjective",
    "text": "The room is very bright today.",
    "correction": "The room is very bright today.",
    "meaning": 4,
    "feedback": "Natural sentence with the correct meaning.",
}


class BatchEncodingStub(dict):
    """The mapping returned by modern Transformers chat templates."""

    def to(self, _device: torch.device) -> BatchEncodingStub:
        return self


class StubTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, _messages, **_kwargs):  # type: ignore[no-untyped-def]
        return BatchEncodingStub(
            input_ids=torch.tensor([[11, 12]]),
            attention_mask=torch.tensor([[1, 1]]),
        )

    def decode(self, _ids, **_kwargs):  # type: ignore[no-untyped-def]
        return json.dumps({key: ROW[key] for key in ("correction", "meaning", "feedback")})


class StubModel:
    device = torch.device("cpu")

    def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        assert not args, "generation inputs must be passed by keyword"
        input_ids = kwargs["input_ids"]
        assert torch.equal(kwargs["attention_mask"], torch.ones_like(input_ids))
        return torch.cat((input_ids, torch.tensor([[13]])), dim=-1)


def test_predict_accepts_batch_encoding_from_chat_template() -> None:
    config = load_config(overrides=["train.thinking=off", "eval.max_new_tokens=4"])
    band_config = BandConfig.from_json(default_config_path())

    predictions = predict_rows(
        config,
        [ROW],
        model=StubModel(),
        tokenizer=StubTokenizer(),
        band_config=band_config,
        max_retries=0,
    )

    assert predictions[0]["prediction"]["meaning"] == 4
