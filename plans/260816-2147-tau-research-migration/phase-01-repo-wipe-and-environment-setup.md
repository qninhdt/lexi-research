---
phase: 1
title: "Repo Wipe & Environment Setup"
status: completed
priority: P1
dependencies: []
---

# Phase 01: Repo Wipe & Environment Setup

## Overview
Purge all legacy files from the old `lexi-research` problem (sentence grading distillation, DVC tracking, and stale submodules) and configure `pyproject.toml` for `tau-research` with Python `>=3.12,<3.14` and modern `uv` workspace settings.

## Requirements
- Functional:
  - Completely wipe `lexi_research/`, `bench/`, `serve/`, `data/`, `notebooks/`, `.dvc/`, `dvc.yaml`, `dvc.lock`, old docs, and old plans.
  - Setup initial package structure under `src/tau_research/` (`__init__.py`).
  - Configure `pyproject.toml` with project name `tau-research`, `requires-python = ">=3.12,<3.14"`, and dependencies for PyTorch, Transformers, TRL, PEFT, Datasets, W&B, and Rich.
- Non-functional:
  - Clean Mypy configuration targeting `src/tau_research` and `tests`.
  - Zero DVC artifacts remaining in repo.
  - GitHub Actions CI workflow updated to test Python 3.12.

## Related Code Files
- Create:
  - `src/tau_research/__init__.py`
  - `configs/sft.yaml`
  - `configs/grpo.yaml`
  - `configs/eval.yaml`
  - `configs/smoke.yaml`
- Modify:
  - `pyproject.toml`
  - `.github/workflows/test.yml`
  - `README.md`
  - `ops/Makefile`
- Delete:
  - `lexi_research/`
  - `bench/`
  - `serve/`
  - `data/`
  - `notebooks/`
  - `.dvc/`, `.dvcignore`, `dvc.yaml`, `dvc.lock`
  - `band_config.json`, `release-manifest.json`
  - `docs/grader-distillation-design.md`, `docs/lexi-lab-design.md`, `docs/results.md`, `docs/reproduction.md`
  - `plans/260729-*`, `plans/260801-*`
  - `tests/*` (except root `conftest.py`)

## Implementation Steps
1. Delete all legacy directories and files specified in the delete list.
2. Update `pyproject.toml`:
   - Rename to `tau-research`.
   - Update `requires-python = ">=3.12,<3.14"`.
   - Define packages under `where = ["src"]`, `include = ["tau_research*"]`.
   - Set dependency groups: `dev` (ruff, mypy, pytest, pytest-asyncio, types-PyYAML), `smoke` (cpu torch, transformers, trl, peft, datasets, accelerate), `colab` (cuda wheels, vllm, flash-attn).
3. Create `src/tau_research/__init__.py` with `__version__ = "0.1.0"`.
4. Create baseline configuration files in `configs/` (`sft.yaml`, `grpo.yaml`, `eval.yaml`, `smoke.yaml`).
5. Update `.github/workflows/test.yml` to target Python 3.12.
6. Regenerate lockfile via `uv lock`.

## Success Criteria
- [x] `uv sync` succeeds without errors on Python 3.12.
- [x] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [x] `uv run mypy` runs against `src/tau_research` and returns no errors.
- [x] Git status confirms no legacy `lexi_research` or `.dvc` files remain.
