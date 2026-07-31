"""Shared fixtures for the format-core tests."""

from __future__ import annotations

import pytest

from lexi_research.format import BandConfig, default_config_path


@pytest.fixture(scope="session")
def config() -> BandConfig:
    """The `band_config.json` that ships with the repo."""
    return BandConfig.from_json(default_config_path())
