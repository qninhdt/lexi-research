"""The model card is generated, and says what it cannot claim.

Two failures this guards against. A hand-edited card drifts from the numbers and
the drift is invisible because both are prose. And a limitations section that
gets paraphrased across revisions turns "fidelity to a teacher" into "accuracy",
which is the one claim this project cannot make.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.report.model_card import LIMITATIONS, ModelCardError, generate, render

REPORT = {
    "stage": "eval",
    "split": "test",
    "rows": 812,
    "ceiling": {"meaning_qwk": 0.82, "correction_edit_f1": 0.74},
    "lineage": {
        "git": {"sha": "a" * 40, "dirty": False},
        "config_sha256": "cfg-hash",
        "dvc_lock_sha256": "lock-hash",
        "libraries": {"torch": "2.13.0"},
    },
    "metrics": {
        "meaning": {
            "qwk": {
                "value": 0.71,
                "reliability": "strong",
                "ceiling": 0.82,
                "fraction_of_ceiling": 0.8658536585365854,
            },
            "exact": {"value": 0.63, "reliability": "strong"},
        },
        "correction": {
            "span_tag_f1": {
                "value": 0.55,
                "reliability": "strong",
                "ceiling": 0.74,
                "fraction_of_ceiling": 0.7432432432432432,
            },
            "span_only_f1": {"value": 0.68, "reliability": "strong"},
        },
        "format": {"validity_rate": {"value": 0.98, "reliability": "strong"}},
    },
    "notes": [],
}

VERDICT = "No. GRPO matched SFT within noise; JEPO and NRT were behind both."


def _card(**kwargs) -> str:
    return render(REPORT, base_model="some/checkpoint", rl_verdict=VERDICT, **kwargs)


def test_the_headline_numbers_come_from_the_report() -> None:
    card = _card()
    assert "0.7100" in card
    assert "0.5500" in card
    assert "812 rows" in card


def test_every_headline_metric_shows_its_share_of_the_ceiling() -> None:
    """0.71 against a ceiling of 0.82 is 87%, and reads very differently to 71%."""
    card = _card()
    assert "86.6%" in card
    assert "74.3%" in card


def test_the_card_states_it_is_fidelity_not_accuracy() -> None:
    card = _card()
    assert "not accuracy against ground truth" in card
    assert "no human gold set" in card.lower()


def test_every_limitation_appears_verbatim() -> None:
    """Paraphrase is how 'fidelity' becomes 'accuracy' over a few revisions."""
    card = _card()
    for limitation in LIMITATIONS:
        assert limitation in card


def test_the_rl_verdict_is_stated_either_way() -> None:
    card = _card()
    assert VERDICT in card


def test_provenance_traces_to_a_commit() -> None:
    card = _card()
    assert "a" * 40 in card
    assert "cfg-hash" in card
    assert "lock-hash" in card


def test_a_dirty_tree_is_admitted() -> None:
    """A result from an uncommitted tree cannot be reproduced from its SHA."""
    dirty = json.loads(json.dumps(REPORT))
    dirty["lineage"]["git"]["dirty"] = True
    card = render(dirty, base_model="x", rl_verdict=VERDICT)
    assert "tree was dirty" in card


def test_the_comparison_table_states_the_axis_that_decides_it() -> None:
    card = _card(
        comparison=[
            {"system": "student", "qwk": 0.71, "e2e_p95_s": 0.8, "cost_per_1k_requests": 0.4},
            {"system": "moe", "qwk": 0.79, "e2e_p95_s": 1.9, "cost_per_1k_requests": 3.2},
        ]
    )
    assert "quality per dollar" in card
    assert "loses for this application" in card


def test_a_report_without_headline_metrics_raises() -> None:
    with pytest.raises(ModelCardError):
        render({"metrics": {}, "lineage": {}, "ceiling": {}}, base_model="x", rl_verdict="n/a")


def test_generation_is_reproducible(tmp_path) -> None:
    """Regenerating must produce no diff, or the card can drift unnoticed."""
    report_path = tmp_path / "eval.json"
    report_path.write_text(json.dumps(REPORT), encoding="utf-8")
    first = generate(report_path, tmp_path / "CARD.md", base_model="x", rl_verdict=VERDICT)
    before = first.read_text(encoding="utf-8")
    generate(report_path, tmp_path / "CARD.md", base_model="x", rl_verdict=VERDICT)
    assert first.read_text(encoding="utf-8") == before


def test_the_card_says_it_is_generated() -> None:
    assert "Do not edit it by hand" in _card()
