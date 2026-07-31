---
phase: 1
title: "DVC pipeline and MLOps spine"
status: done
priority: P1
size: M
dependencies: [0]
---

# Phase 1: DVC pipeline and MLOps spine

## Overview

`dvc.yaml` wired 1 of ~12 stages. Every artifact after `export` was produced by
hand, which means no result could be traced to the code and params that made it.
This phase closes that, adds W&B lineage, ships the Colab launcher, and extends
the smoke gate to cover the data stages.

The MLOps property this phase is after is *lineage*: a number in a report
resolves to a W&B run, which resolves to a DVC stage hash, which resolves to a
commit.

## Requirements

**Functional**

- Every pipeline stage that has code is a DVC stage with declared deps, params,
  outs and metrics.
- Each stage's `cmd` is a `lexi` subcommand — the same command a human types.
- Runs log the DVC lock hash, git SHA and dirty flag, resolved config and its
  hash, library versions and GPU into the run config.
- Adapters and the band config are pushed as a single W&B Artifact. A checkpoint
  without its band config produces meaningless bands (parent design §6), so they
  version together or not at all.
- `notebooks/lexi_colab.ipynb` is generated from a tracked source and contains
  **no cell that defines a function or class**.
- CI runs the smoke gate.

**Non-functional**

- Private Cambridge-derived text stays out of Git and out of W&B.
- The committed notebook carries no execution output and no drift from the CLI.
- `tracking.mode: disabled` imports nothing, which is what CI runs.

## Files

**Created**

- `lexi_research/tracking/lineage.py` — git state, `dvc.lock` and params hashes,
  config hash, library versions, GPU
- `lexi_research/tracking/wandb_run.py` — a run handle that behaves the same
  whether or not anything is recorded
- `lexi_research/data/stages.py` — `sample`, `generate`, `label`, `process`,
  `calibrate` as functions a DVC stage and a human call the same way
- `notebooks/lexi_colab.py` — percent-format source of truth
- `ops/build-notebook.py` — deterministic `.ipynb` emitter, with `--check`
- `tests/tracking/test_lineage.py`, `tests/ops/test_notebook_contract.py`

**Modified**

- `dvc.yaml` — `sample`, `generate`, `label`, `process`, `calibrate`, `sft`
- `lexi_research/cli/__init__.py` — the five data subcommands; every stage
  reports through the tracking handle
- `lexi_research/cli/smoke.py` — the gate now runs the data stages too
- `lexi_research/train/trainer.py` — accepts an open run so Trainer metrics land
  on the same run as the lineage
- `params.yaml` — a `tracking` section
- `ops/Makefile` — `notebook`, `notebook-check`, `repro`, `dag`
- `.env.example` — `WANDB_API_KEY`, `WANDB_PROJECT`
- `pyproject.toml` — the notebook source is excluded from ruff, as it carries
  IPython magics and is checked by its contract test instead

## DVC stage graph

```
export → sample → generate → label → process → calibrate → sft
```

| Stage | cmd | outs | metrics |
|---|---|---|---|
| `sample` | `lexi data sample` | `data/batches/batch_specs.parquet` | senses, batches, spec coverage |
| `generate` | `lexi data generate` | `data/raw/raw_texts.parquet` | distinct-2, reject reasons, cost |
| `label` | `lexi data label` | `data/raw/raw_labels.parquet` | validity rate, reject reasons, cost |
| `process` | `lexi data process` | `data/clean/{processed,rejects,train,val,test,test_strict}.parquet` | reject reasons, distribution before/after balance, leak check |
| `calibrate` | `lexi data calibrate` | `data/clean/band_config.json` | thresholds, band distribution |
| `sft` | `lexi train sft` | `runs/sft/adapter` | train loss |

`generate` and `label` reach a teacher endpoint and cost money; they are declared
so lineage is complete and never run in CI, where `lexi smoke` substitutes the
fixture.

## Deviations from the plan, and why

- **`validate`, `balance` and `split` are one `process` stage.** `process_rows` is
  a single transformation whose intermediates nothing consumes, and its report
  already breaks out reject reasons, the distribution before and after balancing,
  and the contamination check. Three stages would have bought three parquet
  round-trips and no traceability.
- **`calibrate` writes `data/clean/band_config.json`, not the repo-root file.**
  The root `band_config.json` holds the design guesses with `calibrated: false`
  and is a dependency of `process`; if calibration also wrote it the graph would
  be a cycle. The calibrated copy is the pipeline artifact, and it is what ships
  with a checkpoint.
- **`eval` and `bench` are not yet stages.** `dvc.yaml`'s standing rule is that a
  stage is listed only once its entrypoint exists, so `dvc repro` never references
  a missing command. They land with phases 2 and 5.
- **The CI smoke job landed in phase 0**, because the collation defect that phase
  found was invisible without it.

## Tests

| Test | Asserts |
|---|---|
| `test_notebook_contract.py::test_no_definitions` | no cell defines a function or a class |
| `test_notebook_contract.py::test_committed_matches_source` | regenerating from the source yields the committed `.ipynb` byte-for-byte |
| `test_notebook_contract.py::test_no_outputs` | every cell's `outputs` is empty and `execution_count` is null |
| `test_notebook_contract.py::test_the_builder_carries_no_clock_or_random_ids` | `make notebook` twice produces no diff |
| `test_lineage.py::test_lineage_has_required_keys` | git SHA, config hash, library versions, GPU all present |
| `test_lineage.py::test_the_config_hash_moves_with_an_override` | two sweep arms cannot share a config hash |
| `test_lineage.py::test_disabled_mode_records_nothing_and_imports_nothing` | the mode CI runs needs no wandb |
| `test_lineage.py::test_online_without_a_key_falls_back_to_disabled` | a headless box does not hang in a login flow |
| `lexi smoke` | `process` and `calibrate` run over the fixture on every CI run |

## Acceptance

- [x] `dvc dag` renders the full graph.
- [x] Every stage `cmd` is a `lexi` subcommand.
- [x] `make -f ops/Makefile notebook` is idempotent; `notebook-check` passes.
- [x] The smoke gate runs `process` and `calibrate`, so the pure data stages are
      exercised on every PR.
- [x] `uv run pytest` green — 529.
- [ ] `dvc repro` end to end — needs the private source and a teacher endpoint;
      the stages it would run are exercised individually by the gate.
- [ ] A W&B run page showing lineage — needs a key; `tracking.mode` falls back to
      disabled without one, which is what CI exercises.

## Findings

- **`SplitResult.contamination` is never empty.** It carries the split counts
  alongside `cross_split_text_hashes`, so a truthiness check on it would have
  failed every run. The gate now reads the count.
- **Calibration cut points ascend with the penalty, not against it.** Band 4 is
  the cleanest fifth, so its cut is the *lowest* quantile. Reversed, `BandConfig`
  rejects the config outright — which is the validator earning its place.
- **Balancing is doing real work on the fixture**: 50 clean rows become 28 after
  the 15% stratum cap. Visible in the report as `before_balance`/`after_balance`.

## Risks

| Risk | Handling |
|---|---|
| DVC stages that need a teacher endpoint block CI | Declared but never run there; the gate uses the fixture |
| The W&B disabled path rots because it is never exercised | `tracking.mode` falls back to disabled without a key, so CI runs exactly that path on every PR |
| The notebook drifts from the CLI after a flag rename | `test_committed_matches_source` fails, and a renamed flag forces regeneration |
| `dvc repro` has never been run end to end | Each stage is exercised on its own; the first full run is the first real dataset build, and it is the one to watch |
