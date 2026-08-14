"""Prompt hashing, rendering, and the untrusted-input boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from lexi_research.format import MAX_BAND, MIN_BAND, TAGS
from lexi_research.teacher import (
    DiversifySpec,
    SenseRef,
    prompt_hash,
    render_diversify_prompt,
    render_grader_prompt,
    template_names,
)
from lexi_research.teacher.registry import PROMPTS_DIR


def test_all_four_templates_are_present() -> None:
    assert template_names() == (
        "diversify_system.jinja",
        "diversify_user.jinja",
        "grader_system.jinja",
        "grader_user.jinja",
    )


def test_prompt_hash_is_stable_across_calls() -> None:
    assert prompt_hash() == prompt_hash()


def test_prompt_hash_changes_when_a_template_changes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """DVC invalidates the generation stage off this hash, so it must move."""
    before = prompt_hash()
    for name in template_names():
        (tmp_path / name).write_bytes((PROMPTS_DIR / name).read_bytes())
    (tmp_path / "grader_system.jinja").write_text("edited", encoding="utf-8")

    monkeypatch.setattr("lexi_research.teacher.registry.PROMPTS_DIR", tmp_path)
    assert prompt_hash() != before


def test_grader_prompt_is_two_messages(sense: SenseRef) -> None:
    messages = render_grader_prompt("bright", sense, "The room is bright.")
    assert [m["role"] for m in messages] == ["system", "user"]


def test_grader_prompt_carries_the_full_taxonomy(sense: SenseRef) -> None:
    """The student is trained to emit these tags, so all 16 must be defined."""
    system = render_grader_prompt("bright", sense, "x")[0]["content"]
    for tag in TAGS:
        assert f"`{tag}`" in system


def test_grader_prompt_anchors_every_meaning_band(sense: SenseRef) -> None:
    """A rubric without anchors cannot be applied consistently."""
    system = render_grader_prompt("bright", sense, "x")[0]["content"]
    table = system[system.index("# Field 2") :]
    for band in range(MIN_BAND, MAX_BAND + 1):
        assert f"| {band} |" in table


def test_grader_prompt_documents_all_three_operations(sense: SenseRef) -> None:
    system = render_grader_prompt("bright", sense, "x")[0]["content"]
    assert "[A>B:tag]" in system
    assert "[A>:tag]" in system
    assert "[>B:tag]" in system


def test_grader_prompt_forbids_editing_unmarked_text(sense: SenseRef) -> None:
    """Validator check 3 rejects it; the prompt must ask for it in the first place."""
    system = render_grader_prompt("bright", sense, "x")[0]["content"]
    assert "not marked" in system or "untouched" in system


def test_grader_prompt_is_deterministic(sense: SenseRef) -> None:
    """Byte-identical rendering is what the parity guarantee rests on."""
    a = render_grader_prompt("bright", sense, "hello")
    b = render_grader_prompt("bright", sense, "hello")
    assert a == b


def test_grader_user_prompt_carries_target_sense_and_text(sense: SenseRef) -> None:
    user = render_grader_prompt("bright", sense, "The room is bright.")[1]["content"]
    assert "bright" in user
    assert sense.definition in user
    assert sense.pos in user
    assert "The room is bright." in user


def test_diversify_prompt_lists_every_spec(sense: SenseRef) -> None:
    specs = [
        DiversifySpec(spec_id=f"s{i}", profile_id="p", meaning_req=i % 5, error_spec="one")
        for i in range(6)
    ]
    user = render_diversify_prompt("bright", sense, specs, {"p": "omits articles"})[1]["content"]
    for spec in specs:
        assert spec.spec_id in user


def test_diversify_prompt_fails_loudly_on_an_unknown_profile(sense: SenseRef) -> None:
    """A missing profile must not render as an empty trait description."""
    specs = [DiversifySpec(spec_id="s1", profile_id="ghost", meaning_req=4, error_spec="none")]
    with pytest.raises(KeyError):
        render_diversify_prompt("bright", sense, specs, {"p": "omits articles"})
