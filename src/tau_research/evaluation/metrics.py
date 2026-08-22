"""Statistical calculations, bootstrap confidence intervals, and paired deltas."""

from __future__ import annotations

import numpy as np


def calculate_pass_rate(task_results: dict[str, list[float]]) -> float:
    """Calculates overall Pass^1 rate across all recorded trials."""
    all_scores: list[float] = []
    for trials in task_results.values():
        all_scores.extend(trials)
    if not all_scores:
        return 0.0
    return float(np.mean(all_scores))


def task_level_scores(task_results: dict[str, list[float]]) -> dict[str, float]:
    """Mean score per task (for task-level bootstrap / paired deltas)."""
    return {tid: float(np.mean(trials)) if trials else 0.0 for tid, trials in task_results.items()}


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculates mean and bootstrap empirical confidence intervals.

    Callers should pass task-level means (not raw trials) when estimating
    Pass^1 uncertainty across tasks.
    """
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))

    rng = np.random.default_rng(seed)
    boot_means: list[float] = []
    n = len(arr)

    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - confidence_level) / 2.0
    low_ci = float(np.percentile(boot_means, alpha * 100))
    high_ci = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return mean_val, low_ci, high_ci


def compute_paired_deltas(
    base_task_scores: dict[str, float],
    sft_task_scores: dict[str, float],
    rl_task_scores: dict[str, float],
) -> tuple[float, float]:
    """Computes paired mean improvement deltas across shared tasks:

    delta_sft = SFT - Base
    delta_rl = SFT+RL - SFT
    """
    common_tasks = (
        set(base_task_scores.keys())
        .intersection(set(sft_task_scores.keys()))
        .intersection(set(rl_task_scores.keys()))
    )

    if not common_tasks:
        return 0.0, 0.0

    base_avg = np.mean([base_task_scores[t] for t in common_tasks])
    sft_avg = np.mean([sft_task_scores[t] for t in common_tasks])
    rl_avg = np.mean([rl_task_scores[t] for t in common_tasks])

    delta_sft = float(sft_avg - base_avg)
    delta_rl = float(rl_avg - sft_avg)

    return delta_sft, delta_rl
