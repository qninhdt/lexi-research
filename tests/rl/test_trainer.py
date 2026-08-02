"""Integration plumbing for the model-backed RL trainer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lexi_research.cli.config import load_config
from lexi_research.format import BandConfig, default_config_path
from lexi_research.rl.segments import Span
from lexi_research.rl.trainer import token_logprobs, train_rl


def test_loaded_rl_policy_is_wrapped_with_lora_before_optimizer(monkeypatch, tmp_path) -> None:
    """A quantised base policy must not enter RL with all of its weights trainable."""
    import lexi_research.train.trainer as train_module

    class LoadedModel:
        def parameters(self):
            return []

    loaded_model = LoadedModel()
    loaded_tokenizer = object()
    called = False

    def fake_load(*args, **kwargs):
        return loaded_model, loaded_tokenizer

    def fake_attach(model, config):
        nonlocal called
        assert model is loaded_model
        called = True

        class Targets:
            def summary(self):
                return "fake"

        return Targets(), model

    class StopAfterSetup(RuntimeError):
        pass

    def stop_optimizer(*args, **kwargs):
        raise StopAfterSetup

    monkeypatch.setattr(train_module, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(train_module, "attach_adapter", fake_attach)
    monkeypatch.setattr(train_module, "load_rows", lambda path: [])
    monkeypatch.setattr(torch.optim, "AdamW", stop_optimizer)

    config = load_config(
        overrides=[
            "train.thinking=on",
            "train.load_in_4bit=true",
        ]
    )
    with pytest.raises(StopAfterSetup):
        train_rl(
            config,
            train_path="unused.parquet",
            output_dir=tmp_path / "rl",
            band_config=BandConfig.from_json(default_config_path()),
        )

    assert called


def test_token_logprobs_limits_forward_to_the_requested_span() -> None:
    """RL log-probs must not materialise a full-context vocabulary tensor."""
    calls = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            keep = kwargs["logits_to_keep"]
            return SimpleNamespace(logits=torch.zeros((1, keep, 8)))

    result = token_logprobs(Model(), [1, 2, 3, 4, 5, 6], Span(start=3, end=5))

    assert len(result) == 2
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert not args
    assert tuple(kwargs["input_ids"].shape) == (1, 5)
    assert tuple(kwargs["attention_mask"].shape) == (1, 5)
    assert kwargs["logits_to_keep"] == 3
