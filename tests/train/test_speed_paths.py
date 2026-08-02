"""Tests for training optimisations that must preserve the supervised objective."""

from __future__ import annotations

import torch

from lexi_research.train.trainer import (
    _completion_logit_start,
    _drop_unused_multimodal_towers,
    completion_only_loss,
    load_model_and_tokenizer,
)


def test_selective_logits_has_the_same_completion_loss_as_full_logits() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 7, 11)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4, 5, -100, -100],
            [-100, -100, -100, 2, 1, 0, -100],
        ]
    )
    start = _completion_logit_start(labels)

    full = completion_only_loss(logits, labels, start)
    selected = completion_only_loss(logits[:, start:], labels, start)

    assert start == 1
    assert torch.equal(full, selected)


def test_model_loader_selects_sdpa_when_the_declared_class_supports_it(monkeypatch) -> None:
    import transformers

    import lexi_research.train.trainer as trainer_module

    calls = {}

    class Model:
        _supports_sdpa = True

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            return object()

    class Tokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

    monkeypatch.setattr(trainer_module, "resolve_model_class", lambda name: (Model, object()))
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: Tokenizer(),
        raising=False,
    )

    load_model_and_tokenizer("fake/model", load_in_4bit=False)

    assert calls["name"] == "fake/model"
    assert calls["kwargs"]["attn_implementation"] == "sdpa"
    assert calls["kwargs"]["low_cpu_mem_usage"] is True


def test_model_loader_maps_flash_attention_2_to_transformers_capability(monkeypatch) -> None:
    import transformers

    import lexi_research.train.trainer as trainer_module

    calls = {}

    class Model:
        _supports_flash_attn = True

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["kwargs"] = kwargs
            return object()

    class Tokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

    monkeypatch.setattr(trainer_module, "resolve_model_class", lambda name: (Model, object()))
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: Tokenizer(),
        raising=False,
    )

    load_model_and_tokenizer(
        "fake/model",
        load_in_4bit=False,
        attn_implementation="flash_attention_2",
    )

    assert calls["kwargs"]["attn_implementation"] == "flash_attention_2"


def test_text_only_loader_drops_media_towers_without_touching_language_model() -> None:
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Linear(2, 2)
            self.visual = nn.Linear(2, 2)
            self.vision_tower = nn.Linear(2, 2)
            self.audio_tower = nn.Linear(2, 2)
            self.embed_vision = nn.Linear(2, 2)
            self.embed_audio = nn.Linear(2, 2)

    class Model:
        def __init__(self) -> None:
            self.model = Wrapper()

    model = Model()
    removed = _drop_unused_multimodal_towers(model)

    assert removed == (
        "model.visual",
        "model.vision_tower",
        "model.audio_tower",
        "model.embed_vision",
        "model.embed_audio",
    )
    assert model.model.language_model.in_features == 2
    assert all(
        getattr(model.model, name) is None
        for name in (
            "visual",
            "vision_tower",
            "audio_tower",
            "embed_vision",
            "embed_audio",
        )
    )


def test_liger_and_selective_logits_together_are_refused(tmp_path) -> None:
    """Both replace the loss, and a silent winner would change the objective."""
    import json

    import pytest

    from lexi_research.cli.config import load_config
    from lexi_research.train.trainer import TrainerSetupError, train_sft

    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps(
            {
                "target": "bright",
                "definition": "full of light",
                "pos": "adjective",
                "text": "The room is bright.",
                "correction": "The room is bright.",
                "meaning": 4,
                "feedback": "Good.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_config(
        overrides=[
            "train.use_liger_kernel=true",
            "train.selective_logits=true",
        ]
    )

    with pytest.raises(TrainerSetupError, match="both replace the"):
        train_sft(
            config,
            train_path=rows,
            output_dir=tmp_path / "out",
            model=object(),
            tokenizer=object(),
        )
