# The acceptance gate. `make -f ops/Makefile smoke` must be green before any
# real experiment; every target runs from the repo root so paths in params.yaml
# stay relative to it.

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

.PHONY: smoke smoke-gpu lint test fixture

# CPU, no network, no GPU: a randomly-initialised stack over the 50-row fixture.
smoke:
	cd $(ROOT) && uv run --group smoke lexi smoke $(OVERRIDES)

# The same gate against the checkpoint in `train.base_model`. This is the run
# that answers whether PEFT and the quantiser attach to that architecture — a
# tiny stand-in cannot. Install the GPU stack first (`requirements-colab.txt`);
# the CPU `smoke` group deliberately is not used here.
smoke-gpu:
	cd $(ROOT) && uv run lexi smoke --gpu $(OVERRIDES)

lint:
	cd $(ROOT) && uv run ruff check .
	cd $(ROOT) && uv run ruff format --check .
	cd $(ROOT) && uv run mypy

test:
	cd $(ROOT) && uv run pytest -q

# Rebuild the fixture from its source rows; it is committed, so this only runs
# when the rows change.
fixture:
	cd $(ROOT) && uv run python ops/fixtures/build_smoke_fixture.py
