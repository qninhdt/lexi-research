"""Prompt parity — the property that makes this distillation rather than imitation.

Call 2 grades with the same prompt the finetuned student will see at inference.
If the two ever drift, the student learns a function it will never be asked to
compute, and nothing downstream would notice: the outputs look identical either
way.

Convention cannot protect that, so these tests protect it structurally. One
function renders the grader prompt, no other module loads the grader templates,
and the rendered bytes are stable given a fixed nonce.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lexi_research.teacher import registry
from lexi_research.teacher.registry import (
    GRADER_TEMPLATES,
    PROMPTS_DIR,
    render_grader_prompt,
)
from lexi_research.teacher.schemas import SenseRef

SENSE = SenseRef(definition="full of light", pos="adjective")
PACKAGE_ROOT = Path(registry.__file__).resolve().parents[2]


def test_render_is_deterministic() -> None:
    first = render_grader_prompt("bright", SENSE, "The room is bright.")
    second = render_grader_prompt("bright", SENSE, "The room is bright.")
    assert first == second


def _grader_template_readers() -> set[str]:
    """Modules that name a grader template, other than the registry itself.

    Walks the AST for string constants rather than grepping, so a template name
    split across a format string or a comment does not produce a false hit.
    """
    offenders: set[str] = set()
    for path in (PACKAGE_ROOT / "lexi_research").rglob("*.py"):
        if path.name == "registry.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in GRADER_TEMPLATES:
                    offenders.add(str(path.relative_to(PACKAGE_ROOT)))
    return offenders


def test_only_the_registry_loads_the_grader_templates() -> None:
    """Parity by construction: exactly one code path builds the grading prompt."""
    assert _grader_template_readers() == set()


def test_serving_and_generation_share_one_render_function() -> None:
    """Both consumers resolve to the same function object, not two copies."""
    from lexi_research.teacher import render_grader_prompt as exported

    assert exported is render_grader_prompt


@pytest.mark.parametrize("band", [0, 1, 2, 3, 4])
def test_rubric_anchors_present_for_every_band(band: int) -> None:
    """An unanchored rubric cannot be applied consistently, which poisons labels."""
    system = render_grader_prompt("bright", SENSE, "x", nonce="0")[0]["content"]
    assert f"| {band} |" in system


def test_system_prompt_carries_the_full_taxonomy() -> None:
    from lexi_research.format import TAGS

    system = render_grader_prompt("bright", SENSE, "x", nonce="0")[0]["content"]
    for tag in sorted(TAGS):
        assert f"`{tag}`" in system, f"tag {tag} missing from the grader prompt"


def test_system_prompt_states_the_no_drift_rule() -> None:
    """The rule behind validator check 3 must be in the prompt, not only the check."""
    system = render_grader_prompt("bright", SENSE, "x", nonce="0")[0]["content"].lower()
    assert "exactly" in system
    assert "not marked" in system or "untouched" in system


def test_system_prompt_does_not_ask_for_derived_bands() -> None:
    """`grammar` and `naturalness` are computed; asking for them invites disagreement."""
    system = render_grader_prompt("bright", SENSE, "x", nonce="0")[0]["content"]
    assert '"grammar"' not in system
    assert '"naturalness"' not in system


def test_user_prompt_contains_learner_text() -> None:
    user = render_grader_prompt("bright", SENSE, "The room is bright.")[1]
    assert "The room is bright." in user["content"]
    assert "Target word: bright" in user["content"]


def test_prompt_hash_is_stable_across_calls() -> None:
    assert registry.prompt_hash() == registry.prompt_hash()


def test_prompt_hash_changes_when_a_template_changes(tmp_path, monkeypatch) -> None:
    """A prompt edit must invalidate the data generated under the old prompt."""
    before = registry.prompt_hash()

    staging = tmp_path / "prompts"
    staging.mkdir()
    for path in PROMPTS_DIR.glob("*.jinja"):
        (staging / path.name).write_bytes(path.read_bytes())
    target = staging / "grader_system.jinja"
    target.write_text(target.read_text(encoding="utf-8") + "\nOne more rule.\n", encoding="utf-8")

    monkeypatch.setattr(registry, "PROMPTS_DIR", staging)
    assert registry.prompt_hash() != before


def test_prompt_hash_ignores_file_mtime(tmp_path, monkeypatch) -> None:
    """Hashing bytes, not metadata — otherwise a fresh clone invalidates everything."""
    staging = tmp_path / "prompts"
    staging.mkdir()
    for path in PROMPTS_DIR.glob("*.jinja"):
        copy = staging / path.name
        copy.write_bytes(path.read_bytes())

    monkeypatch.setattr(registry, "PROMPTS_DIR", staging)
    before = registry.prompt_hash()
    for path in staging.glob("*.jinja"):
        path.touch()
    assert registry.prompt_hash() == before


def test_template_names_covers_every_template_on_disk() -> None:
    on_disk = {path.name for path in PROMPTS_DIR.glob("*.jinja")}
    assert set(registry.template_names()) == on_disk
    assert set(GRADER_TEMPLATES) <= on_disk


def test_missing_template_variable_raises_rather_than_rendering_empty() -> None:
    """A prompt that silently loses its rubric still returns plausible labels."""
    from jinja2 import UndefinedError

    template = registry._ENV.from_string("Target: {{ nonexistent_variable }}")
    with pytest.raises(UndefinedError):
        template.render()
