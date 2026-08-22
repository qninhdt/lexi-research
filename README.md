# tau-research

> Specialized Business Customer-Service Assistant via SFT → Agentic Reinforcement Learning on $\tau^3$-bench Retail

[![CI](https://github.com/qninhdt/tau-research/actions/workflows/test.yml/badge.svg)](https://github.com/qninhdt/tau-research/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## 1. Overview & Core Thesis

**tau-research** specializes `Qwen/Qwen3.5-2B` into a stateful, tool-calling customer-service agent using turn-by-turn reasoning SFT and online multi-turn Agentic Reinforcement Learning with Hugging Face TRL and `tau2-bench` v1.0.1.

Our empirical goal is demonstrating:
$$\text{Base} < \text{SFT} < \text{SFT + Agentic RL}$$
on the official held-out $\tau^3$-bench Retail test split using single-GPU compute (1 × NVIDIA L4 24GB on Google Colab Pro).

---

## 2. Key Architecture & Methodology

- **Model**: `Qwen/Qwen3.5-2B` (BF16, LoRA, gradient checkpointing, vLLM colocate with sleep mode).
- **Environment**: `sierra-research/tau2-bench` (pinned release `v1.0.1`), Retail domain.
- **SFT Preprocessing**: Trajectories from `fuvty/tau-bench-synthetic` decomposed into per-turn conversational prompt-completion pairs; previous `<think>` traces are stripped from prompt history to match Qwen3.5 official chat template guidelines. `completion_only_loss = True`.
- **Agentic RL**: Online multi-turn environment rollouts using TRL `GRPOTrainer` with custom `rollout_func`, outcome-based binary reward ($R = R_{\text{DB}} \times R_{\text{COMM}}$), and learnable task variance sampling.
- **Evaluation**: 4 trials per task on held-out test split, reporting Pass$^1$ with 95% bootstrap confidence intervals, paired $\Delta_{\text{SFT}}$ & $\Delta_{\text{RL}}$, and 11-category error taxonomy.

---

## 3. Repository Structure

```text
tau-research/
├── configs/                   # Experiment configs (SFT, GRPO, Eval, Smoke)
├── src/tau_research/          # Core Python package
│   ├── data/                  # Dataset preparation, token profiling, splits
│   ├── tau/                   # Gym environment integration, action parser, rewards
│   ├── training/              # SFTTrainer, GRPOTrainer, LoRA merge, difficulty profiling
│   ├── evaluation/            # Benchmark evaluator, bootstrap metrics, error taxonomy
│   └── logging/               # W&B trajectory logger and custom metrics callbacks
├── tests/                     # Test suite (TDD unit tests & mock rollouts)
├── scripts/                   # Colab automation shell scripts
└── ops/Makefile               # Quality gates (check, test, smoke)
```

---

## 4. Quick Start & Reproduction

### Local Setup
```bash
# Clone and install dependencies
uv sync --dev

# Run quality checks
make check

# Run test suite
make test
```

### Google Colab Pro Setup (1 × L4 GPU)
```bash
# Setup environment & third_party/tau2-bench
bash scripts/setup_colab.sh

# Run end-to-end smoke test
bash scripts/smoke_test.sh

# 1. Run SFT training and merge adapter
bash scripts/run_sft.sh

# 2. Run difficulty profiling & Agentic GRPO
bash scripts/run_grpo.sh

# 3. Run final 4-trial evaluation
bash scripts/run_final_eval.sh
```

---

## 5. License
MIT License.
