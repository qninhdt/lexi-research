"""The inference lab: engine adapters behind one interface, and a load generator.

Quality and latency are never reported apart. A quantisation that gains 40%
throughput while losing 0.1 QWK is a decision, not a win, and a report that
states only the first half makes the decision for the reader.
"""

from .runner import BenchError, BenchResult, Sample, arrival_schedule, cdf, percentile

__all__ = [
    "BenchError",
    "BenchResult",
    "Sample",
    "arrival_schedule",
    "cdf",
    "percentile",
]
