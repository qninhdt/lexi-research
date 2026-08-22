"""Tests for statistical metrics, Pass^1 success rate, and bootstrap confidence intervals."""

import numpy as np

from tau_research.evaluation.metrics import (
    bootstrap_confidence_interval,
    calculate_pass_rate,
    compute_paired_deltas,
)


def test_calculate_pass_rate() -> None:
    # 3 tasks, 2 trials each
    # Task 1: [1, 1], Task 2: [1, 0], Task 3: [0, 0]
    task_results = {
        "t1": [1.0, 1.0],
        "t2": [1.0, 0.0],
        "t3": [0.0, 0.0],
    }
    pass1 = calculate_pass_rate(task_results)
    # Average across all 6 trials = (2 + 1 + 0) / 6 = 0.5
    assert np.isclose(pass1, 0.5)


def test_bootstrap_confidence_interval() -> None:
    values = [1.0] * 50 + [0.0] * 50
    mean_val, low_ci, high_ci = bootstrap_confidence_interval(values, n_bootstraps=500, seed=42)

    assert np.isclose(mean_val, 0.5)
    assert low_ci < mean_val < high_ci
    assert 0.35 <= low_ci <= 0.45
    assert 0.55 <= high_ci <= 0.65


def test_compute_paired_deltas() -> None:
    base_scores = {"t1": 0.4, "t2": 0.5}
    sft_scores = {"t1": 0.6, "t2": 0.7}
    rl_scores = {"t1": 0.8, "t2": 0.8}

    delta_sft, delta_rl = compute_paired_deltas(base_scores, sft_scores, rl_scores)
    assert np.isclose(delta_sft, 0.2)
    assert np.isclose(delta_rl, 0.15)
