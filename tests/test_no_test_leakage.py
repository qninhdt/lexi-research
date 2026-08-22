"""Tests for ensuring zero data leakage between SFT synthetic data and official test tasks."""

from tau_research.data.build_splits import split_task_ids_deterministically
from tau_research.data.validate_dataset import assert_no_leakage


def test_split_task_ids_deterministically() -> None:
    synthetic_tasks = [f"synthetic_task_{i:03d}" for i in range(100)]
    train_ids, val_ids = split_task_ids_deterministically(synthetic_tasks, train_ratio=0.9, seed=42)

    assert len(train_ids) == 90
    assert len(val_ids) == 10
    # Overlap between train and val must be empty
    assert set(train_ids).isdisjoint(set(val_ids))


def test_assert_no_leakage_passes_when_clean() -> None:
    train_tasks = ["syn_001", "syn_002", "syn_003"]
    test_tasks = ["tau_test_001", "tau_test_002"]

    # Should not raise
    assert_no_leakage(train_tasks, test_tasks)


def test_assert_no_leakage_raises_on_overlap() -> None:
    train_tasks = ["syn_001", "tau_test_001", "syn_003"]
    test_tasks = ["tau_test_001", "tau_test_002"]

    import pytest

    with pytest.raises(ValueError, match="Leakage detected"):
        assert_no_leakage(train_tasks, test_tasks)
