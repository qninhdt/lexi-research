"""The load generator and its statistics.

Open-loop by construction: requests are issued on a schedule fixed before the run
starts, not after the previous response arrives. A closed-loop generator with N
workers cannot show a saturation knee — as the server slows, the generator issues
fewer requests, the queue never builds, and the report shows a system that
degrades gracefully right up until production discovers it does not. The queue
depth a real user waits behind only appears if arrivals are independent of
service time.

Percentiles come from the raw samples. At a few thousand requests there is no
reason to approximate, and a p99 from a running digest over that many samples is
not a number worth defending.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class BenchError(ValueError):
    """The benchmark configuration or its samples are unusable."""


@dataclass(frozen=True)
class Sample:
    """One request, timed.

    `ttft` is time to first token and `tpot` time per output token thereafter;
    reporting only end-to-end would hide which half of the pipeline is slow.
    """

    started_s: float
    ttft_s: float
    latency_s: float
    output_tokens: int
    ok: bool = True
    warmup: bool = False

    @property
    def tpot_s(self) -> float:
        """Seconds per output token after the first."""
        if self.output_tokens <= 1:
            return 0.0
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


def arrival_schedule(*, rate_per_s: float, duration_s: float, warmup: int = 0) -> list[float]:
    """Offsets at which requests are issued, warm-up first at t=0.

    Deterministic rather than Poisson: two runs of the same arm must be
    comparable, and an exponential inter-arrival adds variance that the repeat-run
    check would then have to absorb.
    """
    if rate_per_s <= 0:
        raise BenchError("arrival rate must be positive")
    if duration_s <= 0:
        raise BenchError("duration must be positive")
    if warmup < 0:
        raise BenchError("warm-up count cannot be negative")

    interval = 1.0 / rate_per_s
    schedule = [0.0] * warmup
    count = int(duration_s * rate_per_s)
    schedule += [index * interval for index in range(count)]
    return schedule


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile over the raw sorted samples."""
    if not values:
        raise BenchError("no samples to take a percentile of")
    if not 0.0 <= fraction <= 1.0:
        raise BenchError(f"percentile fraction {fraction} outside [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cdf(values: Sequence[float], *, points: int = 100) -> list[dict[str, float]]:
    """The latency CDF, which is what the report plots instead of a mean bar.

    A mean hides the tail, and the tail is the only part a user notices.
    """
    if not values:
        raise BenchError("no samples to build a CDF from")
    return [
        {"p": index / points, "latency_s": percentile(values, index / points)}
        for index in range(points + 1)
    ]


@dataclass
class BenchResult:
    """Everything one arm measured, plus what it was measured on."""

    engine: str
    quantisation: str
    concurrency: int
    samples: list[Sample] = field(default_factory=list)
    skipped: str | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    peak_vram_mb: float | None = None
    cost_per_hour: float = 0.0

    @property
    def measured(self) -> list[Sample]:
        """Warm-up excluded: it measures the cache being cold, not the system."""
        return [sample for sample in self.samples if not sample.warmup and sample.ok]

    def statistics(self, *, slo_s: float | None = None) -> dict[str, Any]:
        if self.skipped:
            return {"skipped": self.skipped}
        measured = self.measured
        if not measured:
            raise BenchError("every request failed or was warm-up; nothing to report")

        latencies = [sample.latency_s for sample in measured]
        ttfts = [sample.ttft_s for sample in measured]
        tpots = [sample.tpot_s for sample in measured if sample.output_tokens > 1]
        span = max(sample.started_s + sample.latency_s for sample in measured) - min(
            sample.started_s for sample in measured
        )
        tokens = sum(sample.output_tokens for sample in measured)
        failures = sum(1 for sample in self.samples if not sample.warmup and not sample.ok)

        payload: dict[str, Any] = {
            "requests": len(measured),
            "warmup_discarded": sum(1 for sample in self.samples if sample.warmup),
            "failures": failures,
            "ttft_p50_s": percentile(ttfts, 0.50),
            "ttft_p95_s": percentile(ttfts, 0.95),
            "tpot_mean_s": sum(tpots) / len(tpots) if tpots else 0.0,
            "e2e_p50_s": percentile(latencies, 0.50),
            "e2e_p95_s": percentile(latencies, 0.95),
            "e2e_p99_s": percentile(latencies, 0.99),
            "tokens_per_s": tokens / span if span > 0 else 0.0,
            "peak_vram_mb": self.peak_vram_mb,
        }
        if slo_s is not None:
            # Goodput, not throughput: requests that arrived too late to be
            # useful are not output.
            within = sum(1 for value in latencies if value <= slo_s)
            payload["slo_s"] = slo_s
            payload["goodput_per_s"] = within / span if span > 0 else 0.0
            payload["slo_attainment"] = within / len(latencies)
        if self.cost_per_hour:
            payload["cost_per_1k_requests"] = (
                self.cost_per_hour / 3600 * span / len(measured) * 1000 if measured else 0.0
            )
        return payload


def pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The arms no other arm beats on both quality and latency.

    Drawn explicitly rather than left to the reader: a scatter of a dozen arms
    invites everyone to pick the point nearest their prior.
    """
    frontier = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and float(other["quality"]) >= float(candidate["quality"])
            and float(other["latency_s"]) <= float(candidate["latency_s"])
            and (
                float(other["quality"]) > float(candidate["quality"])
                or float(other["latency_s"]) < float(candidate["latency_s"])
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda point: float(point["latency_s"]))


def variance_across_runs(runs: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    """Spread of one statistic across repeats. Reported rather than hidden.

    A benchmark that cannot reproduce itself is not measuring the system.
    """
    values = [float(run[key]) for run in runs if key in run]
    if len(values) < 2:
        raise BenchError(f"need at least two runs to report variance in {key!r}")
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {
        "mean": mean,
        "std": spread,
        "relative_std": spread / mean if mean else 0.0,
        "min": min(values),
        "max": max(values),
        "runs": len(values),
    }


__all__ = [
    "BenchError",
    "BenchResult",
    "Sample",
    "arrival_schedule",
    "cdf",
    "pareto_frontier",
    "percentile",
    "variance_across_runs",
]
