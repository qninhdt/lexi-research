"""The load generator and its statistics.

The open-loop property is the one that matters. A closed-loop generator with a
fixed worker count silently backs off as the server slows, so the queue never
builds and the report shows graceful degradation right up until production finds
out otherwise.
"""

from __future__ import annotations

import pytest

from bench.runner import (
    BenchError,
    BenchResult,
    Sample,
    arrival_schedule,
    cdf,
    pareto_frontier,
    percentile,
    variance_across_runs,
)


def test_open_loop_arrival() -> None:
    """Arrivals depend on the configured rate alone, never on response latency."""
    schedule = arrival_schedule(rate_per_s=10, duration_s=2)
    assert len(schedule) == 20
    gaps = [b - a for a, b in zip(schedule, schedule[1:], strict=False)]
    assert all(gap == pytest.approx(0.1) for gap in gaps)


def test_the_schedule_is_fixed_before_the_run_starts() -> None:
    """Two calls agree, so a slow server cannot reshape the load it is given."""
    assert arrival_schedule(rate_per_s=4, duration_s=3) == arrival_schedule(
        rate_per_s=4, duration_s=3
    )


def test_warmup_requests_are_issued_first() -> None:
    schedule = arrival_schedule(rate_per_s=5, duration_s=1, warmup=3)
    assert schedule[:3] == [0.0, 0.0, 0.0]
    assert len(schedule) == 8


def test_a_nonpositive_rate_raises() -> None:
    for kwargs in ({"rate_per_s": 0}, {"rate_per_s": -1}):
        with pytest.raises(BenchError):
            arrival_schedule(duration_s=1, **kwargs)


def test_percentiles_exact() -> None:
    """Hand-computed over 1..100: p50 = 50.5, p95 = 95.05, p99 = 99.01."""
    values = list(range(1, 101))
    assert percentile(values, 0.50) == pytest.approx(50.5)
    assert percentile(values, 0.95) == pytest.approx(95.05)
    assert percentile(values, 0.99) == pytest.approx(99.01)
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 100


def test_percentiles_do_not_depend_on_input_order() -> None:
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(values, 0.5) == pytest.approx(3.0)


def test_a_percentile_of_nothing_raises() -> None:
    with pytest.raises(BenchError):
        percentile([], 0.5)


def test_the_cdf_is_monotone() -> None:
    """It is what the report plots, because a mean hides the tail."""
    points = cdf([1.0, 2.0, 3.0, 10.0], points=10)
    latencies = [point["latency_s"] for point in points]
    assert latencies == sorted(latencies)
    assert points[0]["p"] == 0.0 and points[-1]["p"] == 1.0


def _result(**kwargs) -> BenchResult:
    return BenchResult(engine="hf", quantisation="bf16", concurrency=1, **kwargs)


def test_warmup_excluded() -> None:
    samples = [
        Sample(started_s=0.0, ttft_s=5.0, latency_s=9.0, output_tokens=10, warmup=True),
        Sample(started_s=1.0, ttft_s=0.1, latency_s=0.5, output_tokens=10),
        Sample(started_s=2.0, ttft_s=0.1, latency_s=0.5, output_tokens=10),
    ]
    stats = _result(samples=samples).statistics()
    assert stats["requests"] == 2
    assert stats["warmup_discarded"] == 1
    # The 9-second cold start would have dominated every percentile.
    assert stats["e2e_p95_s"] == pytest.approx(0.5)


def test_failures_are_counted_not_averaged_in() -> None:
    samples = [
        Sample(started_s=0.0, ttft_s=0.1, latency_s=0.4, output_tokens=8),
        Sample(started_s=1.0, ttft_s=0.0, latency_s=30.0, output_tokens=0, ok=False),
    ]
    stats = _result(samples=samples).statistics()
    assert stats["requests"] == 1
    assert stats["failures"] == 1


def test_ttft_and_tpot_are_reported_separately() -> None:
    """End-to-end alone would hide which half of the pipeline is slow."""
    sample = Sample(started_s=0.0, ttft_s=0.2, latency_s=1.2, output_tokens=11)
    assert sample.tpot_s == pytest.approx(0.1)
    stats = _result(samples=[sample]).statistics()
    assert stats["ttft_p50_s"] == pytest.approx(0.2)
    assert stats["tpot_mean_s"] == pytest.approx(0.1)


def test_goodput_counts_only_requests_inside_the_slo() -> None:
    """Throughput counts work done; goodput counts work that arrived in time."""
    samples = [
        Sample(started_s=float(i), ttft_s=0.1, latency_s=latency, output_tokens=4)
        for i, latency in enumerate([0.5, 0.5, 3.0, 3.0])
    ]
    stats = _result(samples=samples).statistics(slo_s=1.0)
    assert stats["slo_attainment"] == pytest.approx(0.5)
    assert stats["goodput_per_s"] < stats["tokens_per_s"]


def test_a_skipped_arm_is_reported_as_skipped() -> None:
    """Never absent: a missing arm reads as an arm that was not worth running."""
    stats = _result(skipped="engine has no FP8 on this card").statistics()
    assert stats == {"skipped": "engine has no FP8 on this card"}


def test_an_arm_where_everything_failed_raises() -> None:
    samples = [Sample(started_s=0.0, ttft_s=0.0, latency_s=1.0, output_tokens=0, ok=False)]
    with pytest.raises(BenchError):
        _result(samples=samples).statistics()


def test_pareto_frontier() -> None:
    """Hand-computed: B is dominated by A on both axes and drops out."""
    points = [
        {"name": "A", "quality": 0.80, "latency_s": 1.0},
        {"name": "B", "quality": 0.75, "latency_s": 1.5},
        {"name": "C", "quality": 0.85, "latency_s": 2.0},
        {"name": "D", "quality": 0.70, "latency_s": 0.5},
    ]
    assert [point["name"] for point in pareto_frontier(points)] == ["D", "A", "C"]


def test_an_arm_that_ties_on_both_axes_is_kept_once() -> None:
    points = [
        {"name": "A", "quality": 0.8, "latency_s": 1.0},
        {"name": "B", "quality": 0.8, "latency_s": 1.0},
    ]
    assert len(pareto_frontier(points)) == 2


def test_variance_across_runs_is_reported_rather_than_hidden() -> None:
    runs = [{"e2e_p95_s": 1.0}, {"e2e_p95_s": 1.1}, {"e2e_p95_s": 0.9}]
    spread = variance_across_runs(runs, "e2e_p95_s")
    assert spread["mean"] == pytest.approx(1.0)
    assert spread["relative_std"] > 0
    assert spread["runs"] == 3


def test_variance_needs_a_repeat() -> None:
    with pytest.raises(BenchError):
        variance_across_runs([{"e2e_p95_s": 1.0}], "e2e_p95_s")
