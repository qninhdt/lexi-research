"""Config loading and `--override`.

A sweep changes an arm by passing `--override train.lora_r=64`, so two failures
have to be impossible: a typo silently doing nothing, and a value arriving as
the string `"64"` where an int was meant. Both would produce a run that looks
successful and answers the wrong question.
"""

from __future__ import annotations

import dataclasses

import pytest

from lexi_research.cli.config import (
    Config,
    ConfigError,
    default_params_path,
    load_config,
    parse_override,
)

PARAMS = """
train:
  base_model: Qwen/Qwen3.5-4B
  lora_r: 32
  lora_dropout: 0.05
  enable_thinking: true
  target_module_patterns: attn+mlp
eval:
  max_new_tokens: 256
"""


@pytest.fixture()
def params_file(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(PARAMS, encoding="utf-8")
    return path


def test_override_unknown_key_raises(params_file) -> None:
    """A typo in a sweep must crash, not train a run nobody asked for."""
    with pytest.raises(ConfigError, match="lora_rr"):
        load_config(params_file, overrides=["train.lora_rr=64"])


def test_override_unknown_section_raises(params_file) -> None:
    with pytest.raises(ConfigError, match="trian"):
        load_config(params_file, overrides=["trian.lora_r=64"])


def test_override_through_a_scalar_raises(params_file) -> None:
    """`train.lora_r.x` names nothing; treating it as a new key would hide it."""
    with pytest.raises(ConfigError):
        load_config(params_file, overrides=["train.lora_r.x=1"])


def test_override_types(params_file) -> None:
    config = load_config(params_file, overrides=["train.lora_r=64"])
    value = config.get("train.lora_r")
    assert value == 64
    assert isinstance(value, int)
    assert not isinstance(value, str)


def test_override_float_stays_float(params_file) -> None:
    config = load_config(params_file, overrides=["train.lora_dropout=0.1"])
    assert config.get_float("train.lora_dropout") == pytest.approx(0.1)


def test_override_bool_parses_words_not_truthiness(params_file) -> None:
    """`enable_thinking=false` must be False; `bool("false")` is True."""
    config = load_config(params_file, overrides=["train.enable_thinking=false"])
    assert config.get_bool("train.enable_thinking") is False


def test_override_bool_rejects_nonsense(params_file) -> None:
    with pytest.raises(ConfigError):
        load_config(params_file, overrides=["train.enable_thinking=maybe"])


def test_override_int_rejects_nonsense(params_file) -> None:
    with pytest.raises(ConfigError):
        load_config(params_file, overrides=["train.lora_r=big"])


def test_override_string_passes_through(params_file) -> None:
    config = load_config(params_file, overrides=["train.target_module_patterns=all-linear"])
    assert config.get_str("train.target_module_patterns") == "all-linear"


def test_parse_override_requires_a_key_and_a_value() -> None:
    assert parse_override("train.lora_r=64") == ("train.lora_r", "64")
    for bad in ("train.lora_r", "=64", "train.lora_r="):
        with pytest.raises(ConfigError):
            parse_override(bad)


def test_value_containing_equals_survives() -> None:
    assert parse_override("a.b=x=y") == ("a.b", "x=y")


def test_get_unknown_path_raises(params_file) -> None:
    config = load_config(params_file)
    with pytest.raises(ConfigError):
        config.get("train.nope")


def test_typed_getters_reject_the_wrong_type(params_file) -> None:
    config = load_config(params_file)
    with pytest.raises(ConfigError):
        config.get_int("train.base_model")
    with pytest.raises(ConfigError):
        config.get_str("train.lora_r")
    with pytest.raises(ConfigError):
        config.get_bool("train.lora_r")


def test_get_int_rejects_a_bool(params_file) -> None:
    """`bool` is an `int` subclass; `enable_thinking` is not a rank."""
    config = load_config(params_file)
    with pytest.raises(ConfigError):
        config.get_int("train.enable_thinking")


def test_config_is_immutable(params_file) -> None:
    """Frozen after load: nothing downstream can drift from what W&B recorded."""
    config = load_config(params_file)
    with pytest.raises(TypeError):
        config.values["train"]["lora_r"] = 8  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.values = {}  # type: ignore[misc]


def test_as_dict_is_a_detached_copy(params_file) -> None:
    config = load_config(params_file)
    snapshot = config.as_dict()
    snapshot["train"]["lora_r"] = 8
    assert config.get_int("train.lora_r") == 32


def test_section_returns_a_mapping(params_file) -> None:
    config = load_config(params_file)
    assert config.section("eval")["max_new_tokens"] == 256
    with pytest.raises(ConfigError):
        config.section("train.lora_r")


def test_the_repo_params_file_loads() -> None:
    """The shipped `params.yaml` is the contract every stage reads."""
    config = load_config(default_params_path())
    assert isinstance(config, Config)
    assert config.get_str("train.base_model")
    assert config.get_int("train.max_seq_len") > 0
