"""The `lexi` surface.

The operator rule for this repo is that an experiment never requires writing
Python: every stage is a subcommand, and an arm changes through `--override`.
These tests hold that surface in place — including that commands whose phase has
not landed fail rather than exiting 0, since `lexi smoke` chains stages and a
silent stub would turn the gate into a formality.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.cli import build_parser, main
from lexi_research.cli.smoke import check_fixture
from lexi_research.train.trainer import load_rows

FIXTURE = "ops/fixtures/smoke_50.jsonl"


def _print_config(argv: list[str], capsys) -> dict:
    assert main([*argv, "--print-config"]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("group", ["data", "train", "eval", "bench", "serve", "smoke"])
def test_every_group_is_registered(group, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([group, "--help"])
    assert excinfo.value.code == 0
    capsys.readouterr()


def test_override_reaches_a_subcommand(capsys) -> None:
    """The acceptance criterion: a sweep arm changes without a file edit."""
    payload = _print_config(
        ["train", "sft", "--train", FIXTURE, "--output", "out", "--override", "train.lora_r=8"],
        capsys,
    )
    assert payload["train"]["lora_r"] == 8


def test_repeated_overrides_all_apply(capsys) -> None:
    payload = _print_config(
        [
            "train",
            "sft",
            "--train",
            FIXTURE,
            "--output",
            "out",
            "--override",
            "train.lora_r=8",
            "--override",
            "train.thinking=forced-empty",
        ],
        capsys,
    )
    assert payload["train"]["lora_r"] == 8
    assert payload["train"]["thinking"] == "forced-empty"


def test_an_unknown_override_is_rejected_before_anything_runs() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "sft", "--train", FIXTURE, "--output", "out", "--override", "train.rr=8"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [["eval", "run"], ["bench", "run"], ["serve", "up"], ["train", "rl"]],
    ids=["eval", "bench", "serve", "rl"],
)
def test_commands_from_later_phases_exit_non_zero(argv) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_print_config_runs_no_stage(capsys) -> None:
    """`--print-config` is how the surface stays reviewable without side effects."""
    payload = _print_config(["smoke"], capsys)
    assert payload["smoke"]["fixture"] == FIXTURE


def test_the_committed_fixture_still_meets_its_contract() -> None:
    """Every band, both tag groups, a null correction, a multiword, a clean row."""
    summary = check_fixture(load_rows(FIXTURE))
    assert "16/16 tags" in summary


@pytest.mark.parametrize("ablation", ["a2", "a6", "a7"])
def test_every_ablation_key_reaches_the_sweep_command(ablation, capsys) -> None:
    """An arm changes through --override; the sweep is how they are launched."""
    payload = _print_config(
        ["train", "sweep", "--ablation", ablation, "--train", FIXTURE], capsys
    )
    assert payload["train"]["target_modules"]
