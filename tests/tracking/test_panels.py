"""Unit tests for W&B panels and rich tracking visualisations."""

from __future__ import annotations

from unittest.mock import MagicMock

from lexi_research.tracking import panels


def test_qualitative_rows_formatting() -> None:
    predictions = [
        {
            "req_uid": "req_1",
            "text": "He have a book.",
            "gold": {"correction": "He {+has+} a book.", "meaning": 4, "feedback": "Good job."},
            "prediction": {"correction": "He {+has+} a book.", "meaning": 4, "feedback": "Good job."},
            "reasoning": "Grammar error 'have' -> 'has'",
            "retries": 0,
        }
    ]
    rows = panels.qualitative_rows(predictions)
    assert len(rows) == 1
    assert rows[0][0] == "req_1"
    assert rows[0][1] == "He have a book."
    assert rows[0][2] == "He {+has+} a book."
    assert rows[0][9] is True  # valid


def test_disabled_panels_do_not_raise() -> None:
    mock_run = MagicMock()
    mock_run.active = False

    panels.log_qualitative(mock_run, [])
    panels.log_confusion(mock_run, {"a->b": 1})
    panels.log_reliability(mock_run, [{"mean_confidence": 0.9, "accuracy": 0.85, "count": 10}])
    panels.log_per_band(mock_run, {"0": {"accuracy": 0.8, "mae": 0.2, "count": 5}})
    panels.log_eval_overview(mock_run, {"qwk": 0.85, "edit_f1": 0.90})
    panels.log_hardware_summary(mock_run, peak_vram_gb=8.5, throughput_tokens_per_s=120.0)

    assert not mock_run.log.called


def test_active_panels_log_metrics() -> None:
    mock_run = MagicMock()
    mock_run.active = True

    panels.log_eval_overview(mock_run, {"qwk": 0.85, "edit_f1": 0.90})
    assert mock_run.log.called
    assert mock_run.summary.called

    mock_run.reset_mock()
    panels.log_hardware_summary(mock_run, peak_vram_gb=8.5, throughput_tokens_per_s=120.0)
    assert mock_run.log.called
