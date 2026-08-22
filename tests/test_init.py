"""Basic smoke test for package initialization."""

import tau_research


def test_package_version() -> None:
    assert tau_research.__version__ == "0.1.0"
