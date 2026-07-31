"""The harness, against a fixture whose every metric was computed by hand.

This phase gates the ones after it. Phase 4 will very likely produce "RL does not
beat SFT", and that is a result only if the harness measuring it was trusted
before the RL code existed — otherwise a null result and a measurement bug look
identical. So the numbers below were worked out from the fixture on paper.

The fixture holds 8 rows, 6 gold edits and 6 predicted edits, of which 3 match on
span *and* tag and 5 match on span alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexi_research.eval.harness import (
    HarnessError,
    load_ceiling,
    per_band,
    read_predictions,
    require_calibrated,
    score,
)
from lexi_research.format import BandConfig, default_config_path

FIXTURES = Path(__file__).resolve().parents[2] / "ops" / "fixtures"
PREDICTIONS = FIXTURES / "eval_predictions.jsonl"
CEILING = FIXTURES / "eval_ceiling.json"

LINEAGE = {"git": {"sha": "0" * 40}, "config_sha256": "abc", "libraries": {"torch": None}}


@pytest.fixture()
def calibrated() -> BandConfig:
    """The shipped weights, with the calibration flag flipped for the test."""
    payload = json.loads(default_config_path().read_text(encoding="utf-8"))
    payload["calibrated"] = True
    return BandConfig.from_dict(payload)


@pytest.fixture()
def rows() -> list[dict]:
    return read_predictions(PREDICTIONS)


@pytest.fixture()
def report(rows, calibrated):
    return score(
        rows,
        stage="eval",
        split="test",
        lineage=LINEAGE,
        ceiling=load_ceiling(CEILING),
        band_config=calibrated,
        calibration_bins=4,
    )


def test_refuses_uncalibrated_bands() -> None:
    """The shipped thresholds are design guesses; a grade from a guess is a guess."""
    shipped = BandConfig.from_json(default_config_path())
    assert not shipped.calibrated
    with pytest.raises(HarnessError, match="calibrated"):
        require_calibrated(shipped)


def test_requires_ceiling() -> None:
    """Unnormalised numbers invite comparison against perfection, not the teacher."""
    with pytest.raises(HarnessError, match="ceiling"):
        load_ceiling(None)
    with pytest.raises(HarnessError, match="does not exist"):
        load_ceiling("no/such/ceiling.json")


def test_a_ceiling_missing_a_key_raises(tmp_path) -> None:
    path = tmp_path / "ceiling.json"
    path.write_text(json.dumps({"meaning_qwk": 0.8}), encoding="utf-8")
    with pytest.raises(HarnessError, match="correction_edit_f1"):
        load_ceiling(path)


def test_correction_f1_values(report) -> None:
    """3 of 6 predicted and 3 of 6 gold match on span and tag: P = R = F1 = 0.5."""
    correction = report.groups["correction"]
    assert correction["predicted_edits"]["value"] == 6
    assert correction["gold_edits"]["value"] == 6
    assert correction["span_tag_precision"]["value"] == pytest.approx(0.5)
    assert correction["span_tag_recall"]["value"] == pytest.approx(0.5)
    assert correction["span_tag_f1"]["value"] == pytest.approx(0.5)


def test_span_only_beats_span_tag_by_exactly_the_tag_errors(report) -> None:
    """5 of 6 spans found: 5/6. Two of those five carry the wrong tag: 0.4."""
    correction = report.groups["correction"]
    assert correction["span_only_f1"]["value"] == pytest.approx(5 / 6)
    assert correction["tag_error_rate"]["value"] == pytest.approx(0.4)


def test_cross_tier_confusion_counts_only_the_one_that_could_move_a_band(report) -> None:
    """`agr`→`tense` share weight 2; `sp`→`order` are weights 1 and 3."""
    correction = report.groups["correction"]
    assert correction["cross_tier_confusions"]["value"] == 1
    assert correction["cross_tier_confusion_rate"]["value"] == pytest.approx(0.2)
    assert correction["confusion"]["sp->order"] == 1
    assert correction["confusion"]["agr->tense"] == 1


def test_meaning_metrics_against_hand_counts(report) -> None:
    """5 of 8 bands exact, 7 of 8 within one, total absolute error 4 over 8 rows."""
    meaning = report.groups["meaning"]
    assert meaning["exact"]["value"] == pytest.approx(5 / 8)
    assert meaning["within_one"]["value"] == pytest.approx(7 / 8)
    assert meaning["mae"]["value"] == pytest.approx(0.5)


def test_every_headline_metric_carries_its_ceiling(report) -> None:
    for path in ("meaning.qwk", "correction.span_tag_f1"):
        group, key = path.split(".")
        payload = report.groups[group][key]
        assert payload["ceiling"] is not None
        assert payload["fraction_of_ceiling"] == pytest.approx(
            payload["value"] / payload["ceiling"]
        )


def test_per_band_reports_empty_bands_rather_than_omitting_them() -> None:
    """A band with no rows is a fact about the split, not a key to leave out."""
    breakdown = per_band([4, 4, 3], [4, 3, 3])
    assert breakdown["0"] == {"n": 0, "exact": None, "within_one": None}
    assert breakdown["3"]["n"] == 2
    assert breakdown["3"]["exact"] == pytest.approx(0.5)


def test_format_metrics_on_a_fixture_that_is_entirely_valid(report) -> None:
    fmt = report.groups["format"]
    assert fmt["json_parse_rate"]["value"] == 1.0
    assert fmt["validity_rate"]["value"] == 1.0
    assert fmt["strip_identity_rate"]["value"] == 1.0


def test_taxonomy_is_compared_against_the_teacher_not_against_zero(report) -> None:
    taxonomy = report.groups["taxonomy"]
    assert "teacher_other_rate" in taxonomy
    assert isinstance(taxonomy["student_tags"], dict)


def test_calibration_uses_the_confidences_the_fixture_carries(report) -> None:
    assert "calibration" in report.groups
    assert report.groups["calibration"]["n"] == 8
    assert 0.0 <= report.groups["calibration"]["ece"] <= 1.0


def test_calibration_is_skipped_and_said_so_when_confidence_is_absent(rows, calibrated) -> None:
    """Silently omitting ECE would read as a model that has none to report."""
    stripped = [{k: v for k, v in row.items() if k != "meaning_confidence"} for row in rows]
    report = score(
        stripped,
        stage="eval",
        split="test",
        lineage=LINEAGE,
        ceiling=load_ceiling(CEILING),
        band_config=calibrated,
    )
    assert "calibration" not in report.groups
    assert any("calibration skipped" in note for note in report.notes)


def test_predictions_missing_a_required_field_raise(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"req_uid": "x", "text": "hi"}) + "\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="missing"):
        read_predictions(path)


def test_an_empty_predictions_file_raises(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(HarnessError, match="no predictions"):
        read_predictions(path)


def test_scoring_needs_at_least_one_usable_band(calibrated) -> None:
    rows = [
        {
            "req_uid": "x",
            "text": "hi",
            "gold": {"correction": None, "meaning": 3, "feedback": "ok"},
            "prediction": None,
        }
    ]
    with pytest.raises(HarnessError, match="meaning band"):
        score(
            rows,
            stage="eval",
            split="test",
            lineage=LINEAGE,
            ceiling=load_ceiling(CEILING),
            band_config=calibrated,
        )
