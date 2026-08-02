"""Tests that RL inference batching preserves token probabilities."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lexi_research.rl.segments import Span
from lexi_research.rl.trainer import _token_logprobs_batch, token_logprobs_for_spans


class _PositionLogitModel:
    device = torch.device("cpu")

    def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep=None):
        del attention_mask, use_cache
        batch, width = input_ids.shape
        vocab = 13
        logits = torch.arange(width * vocab, dtype=torch.float32).reshape(1, width, vocab)
        logits = logits.expand(batch, -1, -1).clone()
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_batched_token_logprobs_match_single_sequence_forwards() -> None:
    model = _PositionLogitModel()
    ids = ([2, 4, 6, 8, 10, 12, 14], [3, 5, 7, 9, 11, 13])
    spans = (Span(start=3, end=6), Span(start=2, end=5))

    single = [
        token_logprobs_for_spans(model, row, (span,))[0]
        for row, span in zip(ids, spans, strict=True)
    ]
    batched = _token_logprobs_batch(model, ids, [(span,) for span in spans])

    for expected, actual in zip(single, batched, strict=True):
        assert torch.equal(expected, actual[0])
