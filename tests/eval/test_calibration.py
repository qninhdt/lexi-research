import pytest

from lexi_research.eval.calibrate import (
    assert_confusable_weights,
    calibration_report,
    fit_thresholds,
)
from lexi_research.format import BandConfig, default_config_path


def test_current_config_preserves_confusable_weights() -> None:
    assert_confusable_weights(BandConfig.from_json(default_config_path()))


def test_thresholds_are_monotone_and_require_all_reference_bands() -> None:
    assert fit_thresholds([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]) == (0.5, 1.5, 2.5, 3.5)
    with pytest.raises(ValueError, match="absent"):
        fit_thresholds([0], [4])


def test_report_surfaces_other_rate() -> None:
    assert calibration_report([["other"], ["art"]])["other_rate"] == 0.5
