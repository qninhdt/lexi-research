# tau-research

> Specialized Business Customer-Service Assistant via SFT → Agentic Reinforcement Learning on τ²-bench Retail

[![CI](https://github.com/qninhdt/tau-research/actions/workflows/test.yml/badge.svg)](https://github.com/qninhdt/tau-research/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## 1. Overview & Core Thesis

**tau-research** specializes `Qwen/Qwen3.5-2B` into a stateful, tool-calling customer-service agent using reasoning SFT and online multi-turn Agentic RL (TRL GRPO), targeting the single thesis:

$$\text{Base} < \text{SFT} < \text{SFT + Agentic RL}$$

on the official held-out τ²-bench Retail test split with single-GPU compute (1 × NVIDIA L4 24GB, Google Colab Pro).

## 2. Methodology

- **SFT data**: [`inclusionAI/AReaL-tau2-data`](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data) (Apache-2.0) — retail rows with verified success (`correct=1`, `reward=1.0`) and non-empty thinking traces, converted to per-turn prompt/completion records with a single canonical tool-call format.
- **SFT**: TRL `SFTTrainer`, LoRA (r=16, all-linear), `completion_only_loss`, single-pass chat-template rendering so the completion is the exact inference-time suffix.
- **RL**: TRL `GRPOTrainer` with a custom `rollout_func` running real `AgentGymEnv` episodes against the **official τ² Retail train split** (74 tasks). Agent tokens get policy loss; environment/user feedback tokens are masked out via TRL's `env_mask` path.
- **User simulator**: frozen external API model (`gpt-4.1-mini`) for both RL rollouts and every evaluation — never swapped between checkpoints.
- **Evaluation**: official Retail **test** split (40 tasks) × 4 trials; Pass¹ primary, Pass²/Pass⁴ secondary (leaderboard C(s,k)/C(n,k) convention), paired bootstrap 95% CIs for ΔSFT / ΔRL, 11-category error taxonomy.

## 3. Repository Structure

```text
tau-research/
├── configs/                   # sft / grpo / eval / smoke experiment configs
├── src/tau_research/
│   ├── data/                  # AReaL converter, decontamination audit, validation
│   ├── tau/                   # env factory, rollout loop, action parser, rewards, GRPO rollout_func
│   ├── training/              # SFT trainer, GRPO trainer, difficulty profiler, adapter merge
│   ├── evaluation/            # policy loaders, benchmark harness, metrics, error taxonomy
│   └── logging/               # W&B callbacks and trajectory tables
├── tests/                     # pytest suite (fixtures include real AReaL rows)
├── scripts/                   # Colab automation (setup / smoke / sft / grpo / final eval)
└── ops/Makefile               # quality gates (check, test, smoke)
```

## 4. Quick Start

### Local checks
```bash
uv sync --dev
make check   # ruff + mypy strict
make test    # pytest
```

### Colab Pro pipeline (1 × L4)
```bash
bash scripts/setup_colab.sh          # env + third_party/tau2-bench v1.0.1
bash scripts/smoke_test.sh           # CPU smoke gate

# 1. SFT (downloads/converts AReaL data on first run)
AREAL_JSONL=/path/to/tau2_sft_train.jsonl bash scripts/run_sft.sh

# 2. Difficulty profiling + GRPO
bash scripts/run_grpo.sh

# 3. Base vs SFT vs SFT+RL on held-out test split
bash scripts/run_final_eval.sh
```

### CLI
```bash
uv run tau-research convert-areal --input tau2_sft_train.jsonl
uv run tau-research audit-decontamination --input tau2_sft_train.jsonl
uv run tau-research train-sft --config configs/sft.yaml [--dry-run|--max-steps N]
uv run tau-research profile-difficulty --model-path <sft-merged>
uv run tau-research train-grpo --config configs/grpo.yaml [--dry-run]
uv run tau-research evaluate --model-path <ckpt> --tag sft [--policy vllm]
```

## 5. Statistical Reporting

The held-out test split has only 40 tasks, so single-number deltas are noisy.
Final reports use paired per-task deltas with bootstrap CIs
(`artifacts/evaluation/paired_deltas.json`) and Pass^k; expect ΔRL in the
+2–5pp range based on the AReaL paper's 30B results.

## 6. License

MIT License. SFT dataset: Apache-2.0 (AReaL-tau2-data). Benchmark: sierra-research/tau2-bench v1.0.1.
