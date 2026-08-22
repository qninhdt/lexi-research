---
phase: 7
title: "Colab Automation & CPU Smoke Verification"
status: completed
priority: P1
dependencies: [1, 2, 3, 4, 5, 6]
---

# Phase 07: Colab Automation & CPU Smoke Verification

## Overview
Implement reproduction shell scripts and Colab automation entrypoints in `scripts/`, alongside a lightweight CPU smoke gate in `ops/Makefile` to verify that the entire pipeline (data prep, model loading, dummy training step, mock rollout, evaluation) passes seamlessly in local CI and on remote GPU nodes without manual environment intervention.

## Requirements
- Functional:
  - `scripts/setup_colab.sh`: One-command script to clone `tau2-bench` (pinned to `v1.0.1`), install dependencies with `uv sync --extra colab`, configure W&B, and verify GPU availability.
    <!-- Red Team Finding 6 Fix -->
    - **Secret Sanitization**: Securely pass API keys via `os.environ` / shell env (`OPENAI_API_KEY`, `WANDB_API_KEY`) with zero echo/printing to stdout or log files.
  - `scripts/smoke_test.sh`: Run a minimal 1-task end-to-end verification (1 SFT step, 1 Gym rollout, 1 GRPO step, 1 eval task).
  - `scripts/run_sft.sh`: Launch SFT training run with hyperparameter args and auto-merge to standalone model.
  - `scripts/run_grpo.sh`: Launch difficulty profiling and subsequent Agentic RL training with vLLM colocate.
  - `scripts/run_final_eval.sh`: Run 4-trial held-out test evaluation on all checkpoints and print summary report table.
  - `ops/Makefile`: Define fast CPU targets (`check`, `test`, `smoke`, `clean`).
  - Update `README.md` with complete documentation, architecture diagrams, benchmark reproducibility steps, and Hugging Face model card template.
- TDD / Test Suite:
  - CI verification: Run `make smoke` on GitHub Actions CPU runners to ensure full repo integrity.

## Related Code Files
- Create:
  - `scripts/setup_colab.sh`
  - `scripts/smoke_test.sh`
  - `scripts/run_sft.sh`
  - `scripts/run_grpo.sh`
  - `scripts/run_final_eval.sh`
- Modify:
  - `ops/Makefile`
  - `README.md`
  - `.github/workflows/test.yml`

## Implementation Steps
1. Create `scripts/setup_colab.sh`:
   - Check GPU presence via `nvidia-smi`.
   - Setup `uv`, install root repo and clone `third_party/tau2-bench` (branch `v1.0.1`).
   - Run `uv sync --extra colab` and install Tau gym extras.
   - Verify environment variables for API keys without echoing values.
2. Create execution scripts in `scripts/` (`smoke_test.sh`, `run_sft.sh`, `run_grpo.sh`, `run_final_eval.sh`) with strict bash error handling (`set -euo pipefail`).
3. Update `ops/Makefile` with clean targets:
   - `make check`: Ruff lint, Ruff format check, Mypy.
   - `make test`: Pytest unit tests.
   - `make smoke`: End-to-end CPU smoke gate with mock fixtures.
4. Rewrite `README.md` reflecting the new project scope, research narrative, methodology, and step-by-step reproduction instructions.
5. Verify GitHub Actions `.github/workflows/test.yml` executes `make check` and `make smoke` successfully.

## Success Criteria
- [ ] `make check` and `make smoke` pass locally with zero warnings/errors.
- [ ] All shell scripts in `scripts/` are executable and syntax-checked with `shellcheck` / bash sanity.
- [ ] `README.md` provides clear, self-contained reproduction instructions for Google Colab Pro.
