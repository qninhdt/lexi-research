"""Tests for CLI ``--override section.key=value`` handling."""

import pytest

from tau_research.config_overrides import apply_overrides, coerce_value


def test_coerce_value_types() -> None:
    assert coerce_value("true") is True
    assert coerce_value("false") is False
    assert coerce_value("none") is None
    assert coerce_value("42") == 42
    assert coerce_value("1.0e-4") == 1.0e-4
    assert coerce_value("openai/qwen3.8") == "openai/qwen3.8"


def test_apply_overrides_creates_nested_sections() -> None:
    data: dict = {}
    apply_overrides(data, ["train.packing=true", "model.attn_implementation=flash_attention_2"])
    assert data["train"]["packing"] is True
    assert data["model"]["attn_implementation"] == "flash_attention_2"


def test_apply_overrides_existing_section_not_clobbered() -> None:
    data = {"training": {"learning_rate": 1e-4, "seed": 42}}
    apply_overrides(data, ["training.learning_rate=5e-6"])
    assert data["training"]["learning_rate"] == 5e-6
    assert data["training"]["seed"] == 42


def test_apply_overrides_rejects_missing_equals() -> None:
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.packing"])
