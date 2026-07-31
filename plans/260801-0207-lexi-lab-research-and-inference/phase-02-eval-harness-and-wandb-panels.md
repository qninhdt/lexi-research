---
phase: 2
title: "Eval harness and W&B panels"
status: pending
priority: P1
size: M
dependencies: [1]
---

# Phase 2: Eval harness and W&B panels

# This phase gates everything after it

Phase 4 will very likely produce "RL does not beat SFT". That is a publishable
result *only* if the harness measuring it was trusted before the RL code existed.
Built afterwards, a null result is indistinguishable from a measurement bug. So
this phase runs before any real training, and its own correctness is established
against synthetic inputs with known answers.

## Overview

`lexi_research/eval/metrics.py` has QWK and edit-F1 — about a fifth of what the
design calls for. This phase builds the full suite, the report writer, the
teacher-as-judge path for `feedback`, and the W&B panels that make results
readable.

## Requirements

**Functional**

- `lexi eval run --adapter <path|wandb://…> --split {val,test}` produces a report
  JSON containing every metric in design §5, plus lineage.
- Every metric is computed against the **teacher-self-consistency ceiling**, and
  the report states each metric as both an absolute value and a fraction of that
  ceiling. A student at 0.72 QWK against a teacher that scores 0.74 against itself
  is near-perfect distillation, and a report that omits the ceiling hides this.
- `feedback` has no hard metric. The report emits a teacher-as-judge pairwise
  win-rate and chrF, both tagged `"reliability": "weak"` in the JSON. Any consumer
  that prints them must print the tag.
- Band metrics refuse to render while `band_config.json` has `"calibrated": false`
  — the existing guarantee in the parent design, now enforced in the harness.
- W&B panels are defined in code, not clicked in the UI, so they reproduce.

**Non-functional**

- The whole suite runs on CPU given a predictions file, so metric development and
  testing never need a GPU.
- Generation and scoring are separate steps: `lexi eval predict` writes
  predictions, `lexi eval score` reads them. Re-scoring after a metric fix must
  not require re-running the model.

## Files

**Create**

- `lexi_research/eval/harness.py` — orchestration, ceiling normalisation, report assembly
- `lexi_research/eval/correction.py` — span+tag P/R/F1, span-only F1, confusion matrix
- `lexi_research/eval/calibration.py` — ECE, reliability-diagram bins
- `lexi_research/eval/judge.py` — teacher-as-judge pairwise, order-randomised
- `lexi_research/eval/report.py` — JSON schema, weak-metric tagging, markdown rendering
- `lexi_research/tracking/panels.py` — W&B panel and table definitions in code
- `tests/eval/test_correction.py`, `test_calibration.py`, `test_harness.py`, `test_report.py`

**Modify**

- `lexi_research/eval/metrics.py` — keep as-is, re-export through the harness
- `dvc.yaml` — `eval` stage
- `ops/fixtures/` — add a predictions fixture with hand-computed expected metrics

## Metric implementation notes

| Metric | Note |
|---|---|
| span+tag F1 vs span-only F1 | Reporting both separates *found the wrong place* from *found the right place, called it the wrong thing*. Only the second is tolerable, and the parent design's weight-tier property (§5) says confusions within a tier are harmless. Report a third number: confusion rate **across** weight tiers |
| ECE | 10 equal-mass bins, not equal-width — with a skewed band distribution, equal-width bins leave some nearly empty and ECE becomes noise |
| Teacher self-consistency | Read from the pilot gate artifact produced by `data/pilot_gate.py`; the harness fails loudly if absent rather than silently reporting unnormalised numbers |
| Judge win-rate | Present both orders of every pair and discard non-transitive verdicts; report the discard rate, since a high rate means the judge is not discriminating |
| `other` rate | Compared against the teacher's own `other` rate, not against zero. A student matching the teacher's taxonomy gaps is faithful distillation |

## W&B panels (defined in `panels.py`)

- **Training** — loss split into CE and RL terms; reward mean with a std band; KL
  to reference; reasoning-length histogram over steps; LR; grad-norm.
- **Eval** — tag×tag confusion heatmap; reliability diagram; per-band exact and
  within-1 as stacked bars; band distribution before and after balancing.
- **Ablation** — parallel coordinates over A1–A8 against each metric; Pareto
  scatter of quality against latency with the frontier drawn explicitly.
- **Qualitative** — a `wandb.Table` of input, gold, prediction, reasoning, and diff,
  filterable by band and by tag. This is the panel used most while debugging;
  build it first.
- **Inference** — latency CDF rather than mean bars; throughput against
  concurrency with the SLO line drawn; VRAM timeline.

Latency is reported as a CDF because a mean hides the tail, and the tail is the
only part a user notices.

## Implementation steps

1. Predictions file format and `lexi eval predict` / `lexi eval score` split.
2. `correction.py` against a hand-built fixture where every P/R/F1 value was
   computed by hand — tests assert exact values, not "greater than zero".
3. `calibration.py` with equal-mass binning; test against a synthetic set whose
   ECE is analytically known.
4. `harness.py`: load ceiling, run every metric, normalise, assemble.
5. `report.py`: JSON schema with `reliability` tags; markdown renderer for the
   model card.
6. `judge.py` last — it needs a teacher endpoint, so everything else must already
   work offline.
7. `panels.py` and a first W&B run using the fixture, so panels are verified
   before real data exists.

## Tests

| Test | Asserts |
|---|---|
| `test_correction.py::test_exact_f1_values` | hand-computed P/R/F1 on a fixture with known overlaps |
| `test_correction.py::test_span_only_vs_span_tag` | a prediction with correct spans and wrong tags scores 1.0 span-only and <1.0 span+tag |
| `test_correction.py::test_cross_tier_confusion` | a `word`↔`coll` confusion (same tier) is excluded; a `sp`↔`order` confusion (tiers 1 and 3) is counted |
| `test_calibration.py::test_known_ece` | analytic ECE reproduced within 1e-9 |
| `test_calibration.py::test_equal_mass_bins` | bins hold equal counts under a skewed distribution |
| `test_harness.py::test_refuses_uncalibrated_bands` | raises while `band_config.json` has `calibrated: false` |
| `test_harness.py::test_requires_ceiling` | raises when the self-consistency artifact is absent |
| `test_report.py::test_weak_metrics_tagged` | chrF and judge win-rate carry `reliability: weak` |
| `test_report.py::test_report_is_self_contained` | the JSON alone is enough to interpret every number — lineage and ceiling included |

## Acceptance

- `lexi eval score` on the fixture reproduces every hand-computed value.
- A W&B run renders all five panel groups from fixture data.
- The report JSON validates against its schema and is interpretable without W&B.
- Attempting to report bands with `calibrated: false` fails loudly.

## Risks

| Risk | Handling |
|---|---|
| Metrics look right but are subtly wrong | Every test asserts hand-computed exact values; none assert only a range |
| The judge is expensive and slow | Sampled, not exhaustive; sample size is a config value and recorded in the report |
| Panels defined in code diverge from what the UI shows | One fixture run per panel group is inspected once by eye during this phase and then trusted |
