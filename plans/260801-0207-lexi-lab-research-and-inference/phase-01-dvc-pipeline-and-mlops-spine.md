---
phase: 1
title: "DVC pipeline and MLOps spine"
status: pending
priority: P1
size: M
dependencies: [0]
---

# Phase 1: DVC pipeline and MLOps spine

## Overview

`dvc.yaml` wires 1 of ~12 stages. Every artifact after `export` is currently
produced by hand, which means no result can be traced to the code and params that
made it. This phase closes that, adds W&B lineage, ships the Colab launcher, and
puts `lexi smoke` into CI.

The MLOps skill this phase teaches is *lineage*: a number in a report resolves to
a W&B run, which resolves to a DVC stage hash, which resolves to a commit.

## Requirements

**Functional**

- Every pipeline stage is a DVC stage with declared deps, params, outs, metrics.
- Each stage's `cmd` is a `lexi` subcommand — the same command a human types.
- Training and eval runs log to W&B with the DVC stage hash, git SHA, and resolved
  config in the run config.
- Adapters and `band_config.json` are pushed as a single W&B Artifact. A checkpoint
  without its band config produces meaningless bands (parent design §6), so they
  version together or not at all.
- `notebooks/lexi_colab.ipynb` exists, is generated from a tracked source, and
  contains **no cell that defines a function or class**.
- CI gains a smoke job running the full pipeline on the 50-row fixture.

**Non-functional**

- `dvc repro` from a clean checkout reproduces every non-private artifact.
- Private Cambridge-derived text stays out of Git and out of W&B, as today.
- The committed notebook carries no execution output and no drift from the CLI.

## Files

**Create**

- `lexi_research/tracking/wandb_run.py` — run init, lineage capture, artifact push/pull
- `lexi_research/tracking/lineage.py` — collect git SHA, DVC lock hash, library versions, `nvidia-smi`
- `notebooks/lexi_colab.py` — percent-format source of truth
- `ops/build-notebook.py` — emit `.ipynb` from the source; strip outputs
- `tests/tracking/test_lineage.py`, `tests/ops/test_notebook_contract.py`

**Modify**

- `dvc.yaml` — add every stage after `export`
- `params.yaml` — add `tracking.project`, `tracking.entity`, `tracking.mode`
- `.github/workflows/test.yml` — add the smoke job
- `ops/Makefile` — `make notebook`, `make repro`
- `.env.example` — `WANDB_API_KEY`, `WANDB_PROJECT`

## DVC stage graph

```
export → sample → generate → label → validate → balance → split → calibrate
                                                                      │
                                                       ┌──────────────┴───────┐
                                                    sft ─→ rl ─→ eval      bench
```

| Stage | cmd | outs | metrics |
|---|---|---|---|
| `sample` | `lexi data sample` | `data/batches/` | batch coverage |
| `generate` | `lexi data generate` | `data/raw/sentences.parquet` | distinct-2, reject rate |
| `label` | `lexi data label` | `data/raw/labelled.parquet` | teacher self-consistency |
| `validate` | `lexi data validate` | `data/clean/` | reject rate by cause |
| `balance` | `lexi data balance` | `data/balanced.parquet` | stratum shares before/after |
| `split` | `lexi data split` | `data/{train,val,test}.parquet` | leak check by target word |
| `calibrate` | `lexi data calibrate` | `band_config.json` | thresholds, `calibrated: true` |
| `sft` | `lexi train sft` | `runs/sft/adapter/` | train/val loss |
| `rl` | `lexi train rl` | `runs/rl-${algo}/adapter/` | reward, KL |
| `eval` | `lexi eval run` | `reports/eval-*.json` | full metric suite |
| `bench` | `lexi bench run` | `reports/bench-*.json` | latency, throughput |

Stages requiring a teacher endpoint or a GPU are declared but not run in CI;
`lexi smoke` substitutes fixtures for both.

## Implementation steps

1. **Wire stages incrementally, verifying `dvc repro` after each.** A ten-stage
   `dvc.yaml` written in one commit and debugged afterwards costs more than ten
   small commits.
2. **Every `cmd` is a `lexi` subcommand.** No `python -m` paths in `dvc.yaml`. If a
   stage cannot be expressed as a subcommand, the CLI is incomplete — fix the CLI.
3. **Lineage capture.** One function returning a dict: git SHA, dirty flag,
   `dvc.lock` hash for the current stage, resolved config, `torch`/`transformers`/
   `peft`/`trl` versions, GPU name and driver. Logged into `wandb.init(config=…)`
   and written into every report JSON, so a report is interpretable without W&B.
4. **Artifact discipline.** `lexi train sft` pushes one artifact containing the
   adapter, `band_config.json`, the resolved config, and the lineage dict.
   `lexi eval run --adapter wandb://…` pulls it. Local paths remain supported for
   offline work.
5. **Model Registry promotion.** `lexi eval run` writes a verdict; only a run that
   passes the eval gate may be promoted to the registry. The gate is a config
   value, not a hardcoded threshold.
6. **Notebook generation.** `notebooks/lexi_colab.py` in percent format, converted
   by `ops/build-notebook.py`. Cells:
   1. `!git clone` + `!pip install -e .` + colab requirements
   2. secrets from `google.colab.userdata` into env
   3. `!dvc pull`
   4. `!lexi train sft --config params.yaml --override …`
   5. `!lexi eval run --adapter runs/sft/adapter`
   6. artifact push confirmation
   Nothing else. A test enforces it.
7. **CI smoke job.** New job on the existing workflow: `make smoke`, CPU, no
   network, no secrets. Runs on every PR.

## Tests

| Test | Asserts |
|---|---|
| `test_notebook_contract.py::test_no_definitions` | no cell source contains `def ` or `class ` at column 0 |
| `test_notebook_contract.py::test_committed_matches_source` | regenerating from `notebooks/lexi_colab.py` yields the committed `.ipynb` byte-for-byte |
| `test_notebook_contract.py::test_no_outputs` | every cell's `outputs` is empty and `execution_count` is null |
| `test_lineage.py::test_lineage_has_required_keys` | git SHA, config hash, library versions all present |
| `test_lineage.py::test_offline_mode` | with `tracking.mode: disabled`, no network call is attempted and the run still completes |
| CI smoke job | `make smoke` exits 0 |

## Acceptance

- `dvc repro` from a clean checkout regenerates every non-private artifact.
- `dvc dag` renders the full graph above.
- A W&B run page shows git SHA, DVC hash, and resolved config without opening the repo.
- `make notebook` is idempotent; a second run produces no diff.
- CI smoke job green on a PR.

## Risks

| Risk | Handling |
|---|---|
| DVC stages that need a teacher endpoint block CI | Declared but never run in CI; `lexi smoke` uses recorded fixtures, as `tests/data/test_generate_label_fakes.py` already does |
| W&B offline/disabled path rots because it is never exercised | `tracking.mode: disabled` is what CI runs, so it is exercised on every PR |
| Notebook drifts from the CLI after a flag rename | `test_committed_matches_source` fails; a renamed flag forces regeneration |
