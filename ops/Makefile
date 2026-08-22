# Makefile for tau-research
# Quality gates and acceptance targets

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

.PHONY: check lint test smoke clean

check:
	cd $(ROOT) && uv run ruff check .
	cd $(ROOT) && uv run ruff format --check .
	cd $(ROOT) && uv run mypy

lint: check

test:
	cd $(ROOT) && uv run pytest -q

smoke:
	cd $(ROOT) && uv run python -m tau_research.cli smoke --config configs/smoke.yaml

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
