"""Band calibration refuses cut points that erase a band.

`band_of` decrements once per threshold a penalty exceeds, so two equal cut points
skip a band entirely — it stays in the config and can never be assigned. That is
not hypothetical: on the first real corpus 81% of rows carried no `usage` error,
which put the two lowest quantiles both at exactly 0.0.

The guard has to fire *before* anything is written, because a `band_config.json`
saved with duplicate cut points ships with the model and silently mis-reports band
metrics for every run that reads it.
"""

from __future__ import annotations

import pytest

from lexi_research.data.stages import StageError, _assert_distinct_thresholds


class TestDistinctThresholds:
    def test_distinct_cut_points_pass(self) -> None:
        _assert_distinct_thresholds([0.0, 0.4, 0.9, 1.6], zero_share=0.2)

    def test_duplicate_cut_points_raise(self) -> None:
        with pytest.raises(StageError, match="erases a band"):
            _assert_distinct_thresholds([0.0, 0.0, 0.43, 1.11], zero_share=0.52)

    def test_the_message_names_the_zero_mass_that_caused_it(self) -> None:
        """The operator needs to know it is the data, not a bad fraction."""
        with pytest.raises(StageError) as caught:
            _assert_distinct_thresholds([0.0, 0.0, 0.43, 1.11], zero_share=0.814)

        message = str(caught.value)
        assert "81.4%" in message
        assert "0.0" in message

    def test_duplicates_anywhere_are_caught_not_just_the_first_pair(self) -> None:
        with pytest.raises(StageError, match="erases a band"):
            _assert_distinct_thresholds([0.1, 0.5, 1.2, 1.2], zero_share=0.05)

    def test_a_single_threshold_cannot_collide(self) -> None:
        _assert_distinct_thresholds([0.7], zero_share=0.9)

    def test_an_empty_threshold_list_is_vacuously_fine(self) -> None:
        _assert_distinct_thresholds([], zero_share=0.0)


def test_band_of_skips_a_band_when_two_cut_points_are_equal() -> None:
    """The mechanism the guard exists to prevent, asserted directly.

    Without this, a future refactor could make duplicate thresholds harmless and
    the guard would look like unnecessary strictness.
    """
    import dataclasses

    from lexi_research.format import BandConfig, default_config_path

    config = dataclasses.replace(
        BandConfig.from_json(default_config_path()), thresholds=(0.0, 0.0, 0.5, 1.0)
    )

    assigned = {config.band_of(value) for value in (0.0, 0.25, 0.75, 1.5)}

    # Band 3 is unreachable: any penalty above 0.0 already clears two cut points.
    assert 3 not in assigned
    assert {4, 2, 1, 0} == assigned
