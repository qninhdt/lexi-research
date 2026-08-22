---
title: "Migrate Project to tau-research (Tau-Bench SFT to Agentic RL)"
status: completed
created: 2026-08-16
mode: tdd
brainstorm: "docs/tau-research-architecture-and-migration.md"
blockedBy: []
blocks: []
---

# Plan: Migrate to `tau-research` (Tau-Bench SFT → Agentic RL)

## Executive Summary
Migrate the repository from `lexi-research` (sentence grading distillation) into `tau-research` (Specialized Business Customer Service Agent via SFT → Agentic RL on `τ³-bench` Retail). The goal is to establish a rigorous, reproducible post-training pipeline on a single NVIDIA L4 (24GB VRAM) demonstrating $\text{Base} < \text{SFT} < \text{SFT + Agentic RL}$ on the held-out $\tau^3$ Retail test set.

## Architecture & Methodology Highlights
- **Base Model**: `Qwen/Qwen3.5-2B` (BF16 + LoRA + gradient checkpointing + vLLM colocate).
- **Environment**: `sierra-research/tau2-bench` pinned to `v1.0.1` (Python `>=3.12,<3.14`).
- **SFT Dataset & Preprocessing**: `fuvty/tau-bench-synthetic` Retail successful trajectories converted into turn-by-turn conversational prompt-completion pairs. Prior turn `<think>` reasoning is stripped from history to match Qwen3.5 chat template best practices. Loss is computed strictly on target completion (`assistant_only_loss` / `completion_only_loss`).
- **Agentic RL**: TRL `GRPOTrainer` with custom `rollout_func` interacting with `AgentGymEnv` in normal conversational mode with a frozen user simulator. Binary outcome reward ($R = R_{\text{DB}} \times R_{\text{COMM}}$) and empirical difficulty profiling (prioritizing learnable tasks with $1/4, 2/4, 3/4$ success).
- **Evaluation**: Held-out official Retail test split (4 trials/task) reporting Pass$^1$ with 95% bootstrap confidence intervals, paired deltas ($\Delta_{\text{SFT}}$, $\Delta_{\text{RL}}$), and 11-category error taxonomy distribution.

---

## Phases Overview

| Phase | Title | TDD / Quality Gate Focus | Status | Priority | Dependencies |
|---|---|---|---|---|---|
| [Phase 01](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-01-repo-wipe-and-environment-setup.md) | Repo Wipe & Environment Setup | Clean legacy files, setup Python 3.12 `pyproject.toml`, verify CI `check` | completed | P1 | [] |
| [Phase 02](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-02-sft-data-pipeline-and-loss-masking.md) | SFT Data Pipeline & Loss Masking | Unit tests for turn split, label masking (`-100`), no test leakage | completed | P1 | [01] |
| [Phase 03](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-03-tau-env-action-parser-and-rollout-adapter.md) | Tau Env, Action Parser & Rollout Adapter | Unit tests for tool parsing, action execution, mock Gym rollouts | completed | P1 | [01] |
| [Phase 04](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-04-baseline-and-sft-training-pipeline.md) | Baseline & SFT Training Pipeline | Baseline eval harness, SFT trainer, adapter merge utility | completed | P1 | [02, 03] |
| [Phase 05](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-05-grpo-training-and-difficulty-profiling.md) | GRPO Training & Difficulty Profiling | Difficulty profiler, TRL GRPO `rollout_func`, W&B trajectory logger | completed | P1 | [03, 04] |
| [Phase 06](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-06-heldout-evaluation-and-reporting.md) | Held-Out Evaluation & Error Taxonomy | 4-trial evaluator, bootstrap 95% CI calculator, 11-category error classifier | completed | P2 | [04, 05] |
| [Phase 07](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-07-colab-automation-and-smoke-verification.md) | Colab Automation & CPU Smoke Verification | End-to-end CPU smoke gate in Makefile & Colab bash runner scripts | completed | P1 | [01, 02, 03, 04, 05, 06] |

---

## Global Acceptance Criteria
1. `uv run ruff check .` and `uv run ruff format --check .` pass cleanly.
2. `uv run mypy` passes strictly without any missing legacy type ignores.
3. `uv run pytest -q` passes all unit tests for tokenization, masking, rollout, reward, and parsing.
4. Repository contains zero legacy `lexi_research` or sentence grader code, and zero DVC files.
5. End-to-end CPU smoke test runs in CI within < 90 seconds.

---

## Red Team Review

### Session — 2026-08-16
**Findings:** 6 (6 accepted, 0 rejected)  
**Severity Breakdown:** 1 Critical, 4 High, 1 Medium  

| # | Finding | Severity | Disposition | Applied To |
|---|---|---|---|---|
| 1 | vLLM KV Cache & Optimizer Memory Collision on 24GB L4 | Critical | Accept | [Phase 05](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-05-grpo-training-and-difficulty-profiling.md) |
| 2 | Thinking Loop & Truncation Fallback Handling | High | Accept | [Phase 03](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-03-tau-env-action-parser-and-rollout-adapter.md) |
| 3 | Zero-Variance Reward Batch Handling in GRPO | High | Accept | [Phase 05](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-05-grpo-training-and-difficulty-profiling.md) |
| 4 | Token-Level Chat Template Label Masking Verification | High | Accept | [Phase 02](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-02-sft-data-pipeline-and-loss-masking.md) |
| 5 | SQLite DB State Isolation & Reset Integrity | High | Accept | [Phase 03](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-03-tau-env-action-parser-and-rollout-adapter.md) |
| 6 | Secret & API Key Sanitization in Colab Scripts | Medium | Accept | [Phase 07](file:///home/qninh/projects/lexi-research/plans/260816-2147-tau-research-migration/phase-07-colab-automation-and-smoke-verification.md) |

### Whole-Plan Consistency Sweep
- **Decision Delta Applied**: All 6 accepted mitigations (memory bounds, truncation handling, zero-variance batch resampling, token ID label tests, atomic DB reset, and secret sanitization) have been written directly into their respective phase files.
- **Cross-File Coherence**: Verified that `plan.md` and all 7 phase files consistently reference `tau-research`, `src/tau_research/`, Python 3.12, and zero DVC dependencies.
- **Unresolved Contradictions**: 0. The plan is verified and fully consistent.
