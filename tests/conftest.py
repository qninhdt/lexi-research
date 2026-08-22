"""Pytest fixtures for tau-research test suite."""

import pytest


@pytest.fixture
def sample_retail_task_id() -> str:
    return "retail_task_001"
