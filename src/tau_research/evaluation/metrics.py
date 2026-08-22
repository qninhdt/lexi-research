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


def compute_pass_k(task_results: dict[str, list[float]], k: int) -> float:
    """Computes Pass^k: a task counts only if ALL k attempts succeed.

    With n trials and s successes per task the unbiased estimate is
    C(s, k) / C(n, k), averaged over tasks (leaderboard convention).
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    total = 0.0
    counted = 0
    for trials in task_results.values():
        n = len(trials)
        if n < k:
            continue
        s = int(sum(1 for t in trials if t >= 1.0))
        total += _binomial(s, k) / _binomial(n, k)
        counted += 1
    if counted == 0:
        return 0.0
    return total / counted


def _binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    result = 1.0
    for i in range(min(k, n - k)):
        result = result * (n - i) / (i + 1)
    return result


def paired_bootstrap_delta(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for the paired mean delta (B - A) over shared tasks.

    Resampling shared task IDs removes between-task variance, which is the
    dominant noise source when the test split has only ~40 tasks.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    a = np.array([scores_a[t] for t in shared], dtype=np.float64)
    b = np.array([scores_b[t] for t in shared], dtype=np.float64)
    deltas = b - a

    rng = np.random.default_rng(seed)
    n = len(shared)
    boot = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(deltas[idx]))

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "delta": float(np.mean(deltas)),
        "ci_low": float(np.percentile(boot, alpha * 100)),
        "ci_high": float(np.percentile(boot, (1.0 - alpha) * 100)),
    }
