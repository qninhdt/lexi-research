"""ECE and the reliability diagram, against analytically known answers.

ECE is easy to implement plausibly and wrongly — off-by-one binning, weighting by
bin rather than by count, a signed gap instead of an absolute one — and every
variant returns a number in [0, 1] that looks like a calibration error. So these
pin exact values computed by hand, and check the property equal-mass binning was
chosen for.

(`test_calibration.py` covers `eval/calibrate.py`, which fits band cut points.
This file covers `eval/calibration.py`, which measures whether a confident
prediction is more often right.)
"""

from __future__ import annotations

import pytest

from lexi_research.eval.calibration import (
    CalibrationError,
    equal_mass_bins,
    expected_calibration_error,
    reliability,
)


def test_known_ece() -> None:
    """Two bins of five.

    Bin A holds 0.5, 0.6, 0.6, 0.6, 0.7 — mean confidence 0.6, three of five
    right, accuracy 0.6, gap 0.0. Bin B holds five at 0.9 — mean confidence 0.9,
    two of five right, accuracy 0.4, gap 0.5. Count-weighted over ten rows: 0.25.
    """
    confidences = [0.5, 0.6, 0.6, 0.6, 0.7] + [0.9] * 5
    correct = [True, True, True, False, False] + [True, True, False, False, False]
    assert expected_calibration_error(confidences, correct, bins=2) == pytest.approx(0.25, abs=1e-9)


def test_perfect_calibration_is_zero() -> None:
    assert expected_calibration_error([1.0] * 8, [True] * 8, bins=4) == pytest.approx(0.0)


def test_total_miscalibration_is_one() -> None:
    assert expected_calibration_error([1.0] * 8, [False] * 8, bins=4) == pytest.approx(1.0)


def test_equal_mass_bins() -> None:
    """Bins hold equal counts under a distribution equal-width bins would ruin."""
    confidences = [0.01] * 40 + [0.02] * 40 + [0.99] * 20
    grouped = equal_mass_bins(confidences, [True] * 100, bins=10)
    assert len(grouped) == 10
    assert {item.count for item in grouped} == {10}


def test_equal_width_binning_would_have_collapsed_that_case() -> None:
    """Why the design specifies equal mass: 80% of rows fall in one decile."""
    confidences = [0.01] * 40 + [0.02] * 40 + [0.99] * 20
    assert sum(1 for value in confidences if value < 0.1) == 80


def test_bins_are_ordered_by_confidence() -> None:
    grouped = equal_mass_bins([0.9, 0.1, 0.5, 0.3], [True] * 4, bins=2)
    assert grouped[0].upper <= grouped[1].lower


def test_more_bins_than_rows_collapses_rather_than_erroring() -> None:
    grouped = equal_mass_bins([0.5, 0.6], [True, False], bins=10)
    assert sum(item.count for item in grouped) == 2


def test_reliability_reports_the_diagram_and_the_aggregate() -> None:
    payload = reliability([0.2, 0.4, 0.6, 0.8], [False, False, True, True], bins=2)
    assert payload["n"] == 4
    assert payload["accuracy"] == pytest.approx(0.5)
    assert payload["mean_confidence"] == pytest.approx(0.5)
    assert len(payload["bins"]) == 2
    assert set(payload["bins"][0]) == {"lower", "upper", "count", "mean_confidence", "accuracy"}


def test_overconfidence_and_underconfidence_both_count() -> None:
    """The absolute gap, so a timid grader is not scored as well-calibrated."""
    over = expected_calibration_error([0.9] * 4, [True, False, False, False], bins=1)
    under = expected_calibration_error([0.1] * 4, [True, True, True, False], bins=1)
    assert over == pytest.approx(0.65)
    assert under == pytest.approx(0.65)


def test_empty_input_raises() -> None:
    with pytest.raises(CalibrationError):
        expected_calibration_error([], [])


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(CalibrationError):
        expected_calibration_error([0.5], [True, False])
