---
phase: 6
title: "Held-Out Evaluation & Error Taxonomy"
status: completed
priority: P2
dependencies: [4, 5]
---

# Phase 06: Held-Out Evaluation & Error Taxonomy

## Overview
Implement the scientific evaluation pipeline for running rigorous 4-trial benchmark passes on the official held-out τ³-bench Retail test split across Base, SFT, and SFT+RL checkpoints. Includes statistical estimation (95% bootstrap confidence intervals, paired deltas) and an automated 11-category error taxonomy classifier.

## Requirements
- Functional:
  - `evaluate_tau.py`: Evaluation harness running $N$ trials per task on held-out Retail test split with frozen decoding parameters (`temp=0.6`, `top_p=0.95`, `max_turns=8`) and standardized user simulator. Saves full trajectory data to `eval_results.jsonl`.
  - `metrics.py`: Calculate Pass$^1$ success rate, DB success rate, Communicate success rate, 95% bootstrap confidence intervals, and paired deltas ($\Delta_{\text{SFT}} = \text{SFT} - \text{Base}$, $\Delta_{\text{RL}} = \text{RL} - \text{SFT}$).
  - `error_analysis.py`: Classify failure episodes into 11 distinct categories:
    - `A. invalid tool syntax`
    - `B. nonexistent tool`
    - `C. wrong tool`
    - `D. wrong argument`
    - `E. policy violation`
    - `F. missing required communication`
    - `G. incorrect DB mutation`
    - `H. unnecessary repeated read calls`
    - `I. premature final answer`
    - `J. thinking loop / truncation`
    - `K. user misunderstanding`
- TDD / Test Suite:
  - `tests/test_metrics.py`: Test bootstrap confidence interval calculations, delta significance, and aggregator functions on synthetic evaluation records.
  - `tests/test_error_analysis.py`: Test rule-based error classification on known failure trajectories.

## Related Code Files
- Create:
  - `src/tau_research/evaluation/__init__.py`
  - `src/tau_research/evaluation/evaluate_tau.py`
  - `src/tau_research/evaluation/metrics.py`
  - `src/tau_research/evaluation/error_analysis.py`
  - `tests/test_metrics.py`
  - `tests/test_error_analysis.py`
- Modify:
  - `configs/eval.yaml`

## Implementation Steps
1. **Write TDD Tests First**:
   - Write `tests/test_metrics.py` with mock trial arrays to verify bootstrap CI coverage and variance formulas.
   - Write `tests/test_error_analysis.py` feeding simulated failure traces and asserting accurate error category tags.
2. Implement `evaluate_tau.py`:
   - Support checkpoint selection (`base`, `sft`, `sft_rl`).
   - Run parallel / batched rollouts over test tasks.
   - Save structured `eval_results.jsonl`.
3. Implement `metrics.py`:
   - Compute aggregate pass rates, standard errors, and confidence intervals.
   - Format LaTeX / Markdown tables for reporting.
4. Implement `error_analysis.py`:
   - Analyze log trajectories to extract error distribution percentages per checkpoint.

## Success Criteria
- [ ] `uv run pytest tests/test_metrics.py tests/test_error_analysis.py` passes 100%.
- [ ] Evaluation harness generates formatted markdown comparison table ($X < Y < Z$) and full trajectory JSONL artifacts.
- [ ] Error distribution chart displays frequency shifts across Base, SFT, and RL models.
