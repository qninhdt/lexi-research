"""B8 — the three-way comparison, and the axis that decides it.

Raw quality is the wrong axis. A 30B MoE that beats the student on QWK while
costing several times more per request loses for this application, and a system
that misses the latency SLO is not a cheaper option at all.
"""

from __future__ import annotations

import pytest

from bench.compare import CompareError, SystemResult, assemble, quality_per_dollar

STUDENT = SystemResult(
    system="student",
    quality={"meaning.qwk": 0.71, "correction.span_tag_f1": 0.55},
    latency={"e2e_p95_s": 0.8, "tokens_per_s": 240.0},
    cost_per_1k_requests=0.40,
)
MOE = SystemResult(
    system="moe",
    quality={"meaning.qwk": 0.79},
    latency={"e2e_p95_s": 1.6, "tokens_per_s": 90.0},
    cost_per_1k_requests=3.20,
)
TEACHER = SystemResult(
    system="teacher",
    quality={"meaning.qwk": 0.82},
    latency={"e2e_p95_s": 3.4, "tokens_per_s": 40.0},
    cost_per_1k_requests=12.0,
)


def test_the_higher_quality_system_can_still_lose() -> None:
    """The MoE wins on QWK and loses on quality per dollar. That is the finding."""
    report = assemble([STUDENT, MOE, TEACHER], slo_s=2.0, lineage={})
    assert report["best_quality_per_dollar"] == "student"


def test_a_system_that_misses_the_slo_scores_nothing() -> None:
    """Not "good but slow": it cannot serve this product at all."""
    assert quality_per_dollar(TEACHER, slo_s=2.0) == 0.0
    assert quality_per_dollar(TEACHER, slo_s=5.0) == pytest.approx(0.82 / 12.0)


def test_quality_per_dollar_is_hand_computable() -> None:
    assert quality_per_dollar(STUDENT, slo_s=2.0) == pytest.approx(0.71 / 0.40)
    assert quality_per_dollar(MOE, slo_s=2.0) == pytest.approx(0.79 / 3.20)


def test_a_system_with_no_cost_reports_none_rather_than_free() -> None:
    unpriced = SystemResult(
        system="student", quality={"meaning.qwk": 0.7}, latency={"e2e_p95_s": 0.5}
    )
    assert quality_per_dollar(unpriced, slo_s=2.0) is None


def test_a_skipped_system_stays_in_the_report() -> None:
    """Absent would read as not worth running."""
    skipped = SystemResult(system="moe", skipped="48 GB tier was not rented")
    report = assemble([STUDENT, skipped], slo_s=2.0, lineage={})
    systems = {row["system"]: row for row in report["systems"]}
    assert systems["moe"]["skipped"] == "48 GB tier was not rented"


def test_the_report_states_the_axis(*_) -> None:
    report = assemble([STUDENT], slo_s=2.0, lineage={})
    assert "quality per dollar" in report["note"]
    assert "not raw" in report["note"]


def test_an_empty_comparison_raises() -> None:
    with pytest.raises(CompareError):
        assemble([], slo_s=2.0, lineage={})
