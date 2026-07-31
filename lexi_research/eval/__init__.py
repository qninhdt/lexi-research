"""Evaluation: the metric suite, the report, and the ceiling every number is read against.

Generation and scoring are separate — `predict` needs a GPU, everything else runs
on CPU from a predictions file — so a metric fix never costs a re-run of the
model.
"""

from .calibrate import assert_confusable_weights, calibration_report, fit_thresholds
from .calibration import expected_calibration_error, reliability
from .correction import correction_scores, edit_triples, other_rate, tag_distribution
from .harness import HarnessError, load_ceiling, read_predictions, score, score_file
from .metrics import correction_metrics, meaning_metrics
from .report import Report, ReportError, check_self_contained

__all__ = [
    "HarnessError",
    "Report",
    "ReportError",
    "assert_confusable_weights",
    "calibration_report",
    "check_self_contained",
    "correction_metrics",
    "correction_scores",
    "edit_triples",
    "expected_calibration_error",
    "fit_thresholds",
    "load_ceiling",
    "meaning_metrics",
    "other_rate",
    "read_predictions",
    "reliability",
    "score",
    "score_file",
    "tag_distribution",
]
