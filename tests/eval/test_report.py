"""The report contract: self-contained, and honest about what it cannot measure.

Two failures this guards against. A metric printed without its reliability tag
reads as fact, and `feedback` has no verifiable ground truth — chrF measures
surface overlap with one teacher phrasing, and a judge win-rate measures one
model's taste. And a report that needs the repository checked out to interpret is
a report that will be misread six months later, so the lineage travels with it.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.eval.report import (
    WEAK,
    Metric,
    Report,
    ReportError,
    check_self_contained,
    load,
)

LINEAGE = {
    "git": {"sha": "0" * 40, "dirty": False},
    "config_sha256": "abc123",
    "libraries": {"torch": "2.13.0"},
}
CEILING = {"meaning_qwk": 0.82, "correction_edit_f1": 0.74}


def _report() -> Report:
    report = Report(stage="eval", split="test", rows=8, lineage=LINEAGE, ceiling=CEILING)
    report.add_group("meaning", {"qwk": 0.41, "exact": 0.625})
    report.add_group("feedback", {"chrf": 0.63, "n": 8})
    report.add_group("judge", {"judge_win_rate": 0.52, "judge_discard_rate": 0.30})
    return report


def test_weak_metrics_tagged() -> None:
    payload = _report().as_dict()["metrics"]
    assert payload["feedback"]["chrf"]["reliability"] == WEAK
    assert payload["judge"]["judge_win_rate"]["reliability"] == WEAK
    assert payload["judge"]["judge_discard_rate"]["reliability"] == WEAK
    assert payload["meaning"]["qwk"]["reliability"] == "strong"


def test_report_is_self_contained() -> None:
    """The JSON alone is enough to interpret every number in it."""
    payload = _report().as_dict()
    check_self_contained(payload)
    assert payload["ceiling"] == CEILING
    assert payload["lineage"]["git"]["sha"]


def test_a_report_without_lineage_is_rejected() -> None:
    payload = _report().as_dict()
    del payload["lineage"]["config_sha256"]
    with pytest.raises(ReportError, match="config_sha256"):
        check_self_contained(payload)


def test_a_metric_without_a_reliability_tag_is_rejected() -> None:
    payload = _report().as_dict()
    del payload["metrics"]["meaning"]["qwk"]["reliability"]
    with pytest.raises(ReportError, match="reliability"):
        check_self_contained(payload)


def test_ceiling_normalisation_is_applied_only_to_headline_metrics() -> None:
    payload = _report().as_dict()["metrics"]
    assert payload["meaning"]["qwk"]["fraction_of_ceiling"] == pytest.approx(0.41 / 0.82)
    assert "fraction_of_ceiling" not in payload["meaning"]["exact"]


def test_a_zero_ceiling_yields_no_fraction_rather_than_infinity() -> None:
    metric = Metric(name="meaning.qwk", value=0.4, ceiling=0.0)
    assert metric.fraction_of_ceiling is None


def test_markdown_prints_the_reliability_next_to_the_number() -> None:
    """A reader who has to look up whether a metric is trustworthy will not."""
    rendered = _report().markdown()
    assert "feedback.chrf" in rendered
    for line in rendered.splitlines():
        if "chrf" in line:
            assert "weak" in line


def test_markdown_carries_the_notes() -> None:
    report = _report()
    report.notes.append("calibration skipped: no per-row confidence")
    assert "calibration skipped" in report.markdown()


def test_flat_is_loggable_numbers_only() -> None:
    flat = _report().flat()
    assert flat["meaning.qwk"] == 0.41
    assert all(isinstance(value, float) for value in flat.values())


def test_non_numeric_values_pass_through_untagged() -> None:
    """Tag distributions and confusion matrices are data, not measurements."""
    report = Report(stage="eval", split="test", rows=1, lineage=LINEAGE, ceiling=CEILING)
    report.add_group("taxonomy", {"student_tags": {"tense": 3}, "other_rate": 0.1})
    payload = report.as_dict()["metrics"]["taxonomy"]
    assert payload["student_tags"] == {"tense": 3}
    assert payload["other_rate"]["reliability"] == "strong"


def test_round_trip_through_disk(tmp_path) -> None:
    path = _report().write(tmp_path / "eval.json")
    payload = load(path)
    assert payload["rows"] == 8
    assert json.loads(path.read_text(encoding="utf-8")) == payload
