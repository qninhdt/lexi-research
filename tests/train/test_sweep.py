"""Sweep enumeration and resume.

Two properties carry the ablations. An arm is a set of overrides and nothing
else — the moment changing one needs a code edit, the diff between two runs stops
being only the axis under test. And state is written after every arm, because
Colab kills sessions and a sweep that restarts from zero after four of seven arms
costs more GPU-hours than the sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lexi_research.train.sweep import (
    Arm,
    SweepError,
    SweepState,
    arm_names,
    available,
    default_ablation_path,
    iter_arms,
    load_ablation,
    pending,
    summarise,
)

ABLATIONS = Path(__file__).resolve().parents[2] / "ops" / "ablations"


def test_the_three_phase_three_ablations_are_defined() -> None:
    assert set(available()) >= {"a2", "a6", "a7"}


def test_a7_has_the_two_mask_arms() -> None:
    ablation = load_ablation(ABLATIONS / "a7-mask.yaml")
    assert arm_names(ablation) == ["a7-completion-only", "a7-full-sequence"]
    assert ablation.arms[1].overrides == ("train.completion_only=false",)


def test_a2_carries_the_arm_that_makes_the_other_two_interpretable() -> None:
    """Without `forced-empty`, a win for `on` could be the scaffold, not reasoning."""
    ablation = load_ablation(ABLATIONS / "a2-thinking.yaml")
    assert arm_names(ablation) == ["a2-on", "a2-off", "a2-forced-empty"]


def test_arms_enumerated() -> None:
    """A6 is a 3x2 grid plus the legacy name list: seven arms."""
    ablation = load_ablation(ABLATIONS / "a6-lora.yaml")
    assert len(ablation.arms) == 7
    assert "a6-legacy-name-list" in arm_names(ablation)
    assert {"train.lora_r=8", "train.target_modules=all-linear"} == set(
        next(arm for arm in ablation.arms if arm.name == "a6-8-all-linear").overrides
    )


def test_the_legacy_arm_pins_a_pattern_list_not_a_preset() -> None:
    """Its whole point is to be the wrong thing to do on a differently-named stack."""
    ablation = load_ablation(ABLATIONS / "a6-lora.yaml")
    legacy = next(arm for arm in ablation.arms if arm.name == "a6-legacy-name-list")
    assert any("q_proj,k_proj" in override for override in legacy.overrides)


def test_every_arm_of_every_ablation_is_only_overrides() -> None:
    for path in ABLATIONS.glob("*.yaml"):
        for arm in load_ablation(path).arms:
            assert arm.overrides
            for override in arm.overrides:
                assert "=" in override


def test_overrides_render_booleans_the_way_the_config_parses_them() -> None:
    """`bool("False")` is True, so `false` has to reach the parser lowercase."""
    ablation = load_ablation(ABLATIONS / "a7-mask.yaml")
    assert "train.completion_only=false" in ablation.arms[1].overrides


def test_resume_skips_completed(tmp_path) -> None:
    ablation = load_ablation(ABLATIONS / "a6-lora.yaml")
    state = SweepState.load(tmp_path / "state.json")
    for arm in ablation.arms[:2]:
        state.record(arm, {"steps": 10})

    remaining = list(iter_arms(ablation, state))
    assert len(remaining) == len(ablation.arms) - 2
    assert all(not state.is_done(arm) for arm in remaining)


def test_state_survives_a_killed_session(tmp_path) -> None:
    ablation = load_ablation(ABLATIONS / "a7-mask.yaml")
    state = SweepState.load(tmp_path / "state.json")
    state.record(ablation.arms[0], {"steps": 4})

    reopened = SweepState.load(tmp_path / "state.json")
    assert reopened.is_done(ablation.arms[0])
    assert pending(ablation, reopened) == [ablation.arms[1]]


def test_restart_reruns_everything(tmp_path) -> None:
    ablation = load_ablation(ABLATIONS / "a7-mask.yaml")
    state = SweepState.load(tmp_path / "state.json")
    state.record(ablation.arms[0], {"steps": 4})
    assert len(list(iter_arms(ablation, state, resume=False))) == 2


def test_state_records_what_the_arm_actually_ran(tmp_path) -> None:
    """An arm audited later needs its overrides, not just its name."""
    state = SweepState.load(tmp_path / "state.json")
    arm = Arm(name="a6-8-attn", overrides=("train.lora_r=8", "train.target_modules=attn"))
    state.record(arm, {"targets": "attn: 64 modules"})
    reopened = SweepState.load(tmp_path / "state.json")
    assert reopened.completed["a6-8-attn"]["overrides"] == list(arm.overrides)
    assert "targets" in reopened.completed["a6-8-attn"]


def test_summarise_counts_progress(tmp_path) -> None:
    ablation = load_ablation(ABLATIONS / "a2-thinking.yaml")
    state = SweepState.load(tmp_path / "state.json")
    assert "0/3" in summarise(ablation, state)
    state.record(ablation.arms[0], {})
    assert "1/3" in summarise(ablation, state)


def test_default_ablation_path_resolves_by_key() -> None:
    assert default_ablation_path("a6").name == "a6-lora.yaml"
    with pytest.raises(SweepError):
        default_ablation_path("a9")


def test_a_definition_with_no_arms_raises(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("key: a0\nquestion: nothing\n", encoding="utf-8")
    with pytest.raises(SweepError, match="no arms"):
        load_ablation(path)


def test_duplicate_arm_names_raise(tmp_path) -> None:
    """Two arms writing to one directory would silently overwrite each other."""
    path = tmp_path / "dupes.yaml"
    path.write_text(
        "key: a0\nquestion: q\narms:\n"
        "  - name: same\n    overrides: {train.lora_r: 8}\n"
        "  - name: same\n    overrides: {train.lora_r: 16}\n",
        encoding="utf-8",
    )
    with pytest.raises(SweepError, match="duplicate"):
        load_ablation(path)


def test_an_arm_without_overrides_raises(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("key: a0\nquestion: q\narms:\n  - name: only\n", encoding="utf-8")
    with pytest.raises(SweepError, match="overrides"):
        load_ablation(path)


def test_every_override_names_a_real_config_key() -> None:
    """A typo in an arm would otherwise train a run nobody asked for."""
    from lexi_research.cli.config import load_config

    config = load_config()
    for path in ABLATIONS.glob("*.yaml"):
        for arm in load_ablation(path).arms:
            config.with_overrides(arm.overrides)
