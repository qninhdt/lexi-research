# Brainstorm Report: Migration to `tau-research`

**Date**: 2026-08-16  
**Status**: Approved & Agreed  
**Target Project**: `tau-research` (Package: `tau_research`)  
**Domain**: Retail Customer Service Agent (`τ³-bench` v1.0.1)  
**Primary Model**: `Qwen/Qwen3.5-2B`  
**Core Thesis**: $\text{Base} < \text{SFT} < \text{SFT + Agentic RL}$ on held-out $\tau^3$ Retail test split  

---

## 1. Executive Summary & Problem Framing

### 1.1 Objective
Transform the existing repository infrastructure into an end-to-end post-training and evaluation platform for specialized business agents. The goal is to prove that **Agentic RL (online multi-turn environment interaction with verifiable database rewards)** outperforms standard SFT and Base reasoning models on the **τ³-bench Retail** domain on single-GPU hardware (1 × NVIDIA L4 24GB on Colab Pro).

### 1.2 Non-Negotiable Constraints & Boundaries
- **Domain Scope**: **Retail domain ONLY** in v1 (text modality, half-duplex). No Voice, Airline, Banking/RAG, Browser, or Code Execution.
- **Hardware**: Google Colab Pro 1 × NVIDIA L4 24GB (`BF16 base + LoRA + gradient checkpointing + batch size 1 + vLLM colocate for RL`).
- **Dependencies**: `sierra-research/tau2-bench` pinned to `v1.0.1`, Python `>=3.12, <3.14`.
- **Training Stack**: Hugging Face `transformers >= 5.2`, `trl` (`GRPOTrainer` with custom `rollout_func`), `peft`, `vllm`.
- **No Handcrafted Prompts/ChatML**: Always use official `tokenizer.apply_chat_template(enable_thinking=True)` and strip previous turns' `<think>` tokens from multi-turn history per Qwen3.5 official guidelines.

---

## 2. Migration & Repository Cleanup Blueprint

### 2.1 Legacy Files to Delete (Complete Wipe)
1. **Old Package**: `lexi_research/` (all submodules: `teacher`, `format`, `eval`, `report`, `rl`, `train`, `data`, `cli`).
2. **Old Submodules & Benchmarks**: `bench/`, `serve/`, `data/`, `notebooks/`.
3. **MLOps Overkill**: `.dvc/`, `.dvcignore`, `dvc.yaml`, `dvc.lock`, `band_config.json`, `release-manifest.json`.
4. **Old Ops & Scripts**: `ops/ablations/`, `ops/fixtures/`, `ops/tag-distribution-reference.py`.
5. **Old Tests**: All tests in `tests/` (to be replaced with new test suite).
6. **Old Documentation & Plans**: `docs/grader-distillation-design.md`, `docs/lexi-lab-design.md`, `docs/results.md`, `docs/reproduction.md`, `plans/260729-*`, `plans/260801-*`.

### 2.2 Reusable Infrastructure to Retain & Upgrade
1. **Package & Environment Management**: `pyproject.toml` (renamed to `tau-research`, `requires-python = ">=3.12,<3.14"`), `uv.lock`.
2. **Quality Gates & Tooling**:
   - `ruff` (linter & formatter)
   - `mypy` (strict mode with lazy-import overrides for heavy ML frameworks)
   - `pytest` + `pytest-asyncio`
3. **CI Pipeline**: `.github/workflows/test.yml` (two jobs: `check` for fast lint/type/unit checks, `smoke` with CPU training stack).
4. **Ops & Scripts**: Colab GPU runner scripts in `scripts/`.

---

## 3. Target System Architecture (`src/` Layout)

```text
tau-research/
├── configs/
│   ├── sft.yaml               # SFT hyperparameters (lr 1e-4, LoRA r=16 a=32, max_len 4096)
│   ├── grpo.yaml              # GRPO hyperparameters (lr 5e-6-1e-5, G=4, loss_type: dapo)
│   ├── eval.yaml              # Decoding parameters (temp 0.6, top_p 0.95, max_turns 8)
│   └── smoke.yaml             # Quick end-to-end dry run config
│
├── src/
│   └── tau_research/
│       ├── __init__.py
│       ├── data/
│       │   ├── prepare_sft.py       # Converts tau-bench-synthetic trajectories to per-turn format
│       │   ├── build_splits.py      # Splits 90/10 by synthetic task_id (leakage prevention)
│       │   ├── profile_lengths.py   # Token distribution profiler (P50/P90/P95/P99)
│       │   └── validate_dataset.py  # Verifies label mask (-100 on user/tool/prev turns)
│       │
│       ├── tau/
│       │   ├── rollout.py           # Custom multi-turn rollout loop connecting Gym to policy
│       │   ├── action_parser.py     # Parses thinking content and tool call arguments
│       │   ├── reward.py            # Official outcome reward extractor (R = R_DB * R_COMM)
│       │   └── user_simulator.py    # Standardized, frozen user simulator wrapper
│       │
│       ├── training/
│       │   ├── train_sft.py         # SFTTrainer with conversational prompt-completion loss
│       │   ├── train_grpo.py        # GRPOTrainer with custom rollout_func & vLLM colocate
│       │   ├── difficulty.py        # Task success rate profiling (learnable task filtering)
│       │   └── merge_adapter.py     # Base + SFT LoRA merge for clean RL policy start
│       │
│       ├── evaluation/
│       │   ├── evaluate_tau.py      # Official held-out test evaluation harness (4 trials/task)
│       │   ├── metrics.py           # Bootstrap 95% CI, delta computations, success rates
│       │   └── error_analysis.py    # 11-category automatic error taxonomy classifier
│       │
│       └── logging/
│           ├── wandb_callbacks.py   # Metrics and GPU memory tracking
│           └── trajectory_logger.py # Samples and logs W&B episode trajectory tables
│
├── third_party/
│   └── tau2-bench/                  # Git submodule / pinned checkout (v1.0.1)
│
├── tests/
│   ├── conftest.py
│   ├── test_chat_template.py        # Validates thinking tokens & history stripping
│   ├── test_loss_mask.py            # Validates completion-only -100 labels
│   ├── test_tau_rollout.py          # Mocks Gym env and verifies multi-turn loop
│   ├── test_reward.py               # Checks DB & COMMUNICATE reward calculation
│   ├── test_no_test_leakage.py      # Asserts no held-out task IDs in train split
│   └── test_action_parser.py        # Validates tool call schema parsing
│
├── scripts/
│   ├── setup_colab.sh               # Colab environment setup & tau2-bench install
│   ├── smoke_test.sh                # End-to-end 1-task sanity check
│   ├── run_sft.sh                   # SFT training launcher
│   ├── run_grpo.sh                  # Agentic RL training launcher
│   └── run_final_eval.sh            # 4-trial held-out benchmark runner
│
├── pyproject.toml
└── README.md
```

---

## 4. Technical Strategy & Key Methodological Decisions

### 4.1 Per-Turn SFT Preprocessing with History Sanitization
- **Problem**: Full-trajectory SFT (concatenating all turns into a 10k+ token target) wastes compute, causes memory explosion on 24GB GPUs, and re-trains old reasoning traces.
- **Solution**: Decompose an $N$-turn episode into $N$ distinct training examples using conversational prompt-completion pairs.
- **Qwen3.5 Best Practice**: For prompt history in turn $k$, strip previous `<think>` blocks and retain only final assistant tool calls/messages. Target completion contains the current turn's `<think>` reasoning + action.
- **Loss Masking**: Verified unit test ensuring `labels == -100` for system, user, previous assistant actions, and tool execution observations.

### 4.2 Agentic RL via TRL `GRPOTrainer` + Custom `rollout_func`
- **TRL Integration**: Use `rollout_func` instead of `environment_factory` because Tau-Bench's interactive loop involves non-deterministic user simulator replies following plain assistant messages.
- **vLLM Colocation**: Enable `use_vllm=True`, `vllm_mode="colocate"`, `vllm_enable_sleep_mode=True`, and `beta=0.0` to eliminate reference model memory overhead on 24GB VRAM.
- **Sparse Outcome Reward**: Default reward is the official binary task reward ($R \in \{0, 1\} = R_{\text{DB}} \times R_{\text{COMM}}$). No dense intermediate shaping to avoid credit assignment degradation.
- **Variance Optimization**: Run empirical difficulty profiling (4 rollouts per train task with SFT policy) before RL. Sample ~70% *learnable* tasks ($1/4, 2/4, 3/4$ success) to maximize non-zero reward variance for GRPO advantage computation.

### 4.3 Rigorous Statistical Evaluation Protocol
- **Dataset**: `τ³-bench` Retail official held-out test split (4 trials/task).
- **Control**: Identical decoding parameters (`temperature=0.6`, `top_p=0.95`, `max_turns=8`), identical frozen user simulator, and identical benchmark commit across Base, SFT, and SFT+RL checkpoints.
- **Metrics**: Pass$^1$ success rate with 95% bootstrap confidence intervals, paired $\Delta_{\text{SFT}}$ and $\Delta_{\text{RL}}$, DB reward, Communication reward, and 11-category error taxonomy breakdown.

---

## 5. Execution Stages & Acceptance Gates

| Stage | Milestone | Gate Criteria |
|---|---|---|
| **Stage 0** | **Repo Wipe & Infra Setup** | Delete legacy files, update `pyproject.toml` (`tau-research`, Python 3.12), verify CI `check` & `smoke`. |
| **Stage 1** | **Data Pipeline & Unit Tests** | `prepare_sft.py`, `build_splits.py`, `profile_lengths.py`, and test suite pass (leakage check + loss mask tests). |
| **Stage 2** | **Tau Env & Rollout Adapter** | `rollout.py`, `action_parser.py`, `reward.py` verified with mock & real tau2-bench Retail task. |
| **Stage 3** | **Baseline Evaluation** | Base `Qwen3.5-2B` evaluated on held-out Retail test split (Pass$^1$ baseline established). |
| **Stage 4** | **SFT Training & Merge** | SFT LoRA trained on synthetic Retail, validated $\Delta_{\text{SFT}} > 0$, merged to `qwen3.5-2b-tau-retail-sft-merged`. |
| **Stage 5** | **RL Difficulty Profiling & Smoke** | 4-rollout profiling on train split, GRPO 20-step smoke test passes without OOM. |
| **Stage 6** | **Agentic GRPO Training** | GRPO training with learnable-heavy sampling, reward convergence monitored on W&B. |
| **Stage 7** | **Final 4-Trial Benchmark & Report** | 3-way evaluation (Base vs SFT vs SFT+RL), bootstrap CIs computed, error distribution analyzed. |
