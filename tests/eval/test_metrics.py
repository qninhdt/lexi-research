import pytest

from lexi_research.eval.metrics import correction_metrics, meaning_metrics


def test_meaning_metrics_capture_ordinal_agreement() -> None:
    result = meaning_metrics([4, 2], [4, 1])
    assert result["exact"] == 0.5
    assert result["within_one"] == 1.0
    assert result["mae"] == 0.5


def test_correction_metrics_and_length_guard() -> None:
    assert correction_metrics([[(0, 1, "sp")]], [[(0, 1, "sp")]])["edit_f1"] == 1
    with pytest.raises(ValueError):
        meaning_metrics([1], [])
