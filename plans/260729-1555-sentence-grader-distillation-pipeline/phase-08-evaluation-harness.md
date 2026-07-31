---
phase: 8
title: "Evaluation harness"
status: pending
priority: P1
effort: "2d"
dependencies: [6, 7]
---

# Phase 8: Evaluation harness

## Overview

Measure how faithfully the student reproduces the teacher on held-out data, and
report the ceiling that bounds every number. Produces the tables the model card and
writeup cite.

## Requirements

**Functional**

- Run the trained adapter over `test.parquet` and score against teacher labels
- Report `meaning` agreement, `correction` edit-level P/R/F1, format validity,
  derived-band agreement, latency, VRAM
- Report teacher self-consistency as the explicit ceiling
- Slice every metric by band, POS, profile CEFR, sentence length, edit count
- Emit machine-readable results (`parquet`/`json`) plus a markdown comparison

**Non-functional**

- Deterministic: greedy decode, fixed seed, pinned adapter revision
- Test set touched **once**, after the config is locked in Phase 7

## Architecture

### What can and cannot be claimed

There is no human gold. Every number here is **fidelity to the teacher**, not
accuracy. The student cannot exceed the teacher by construction. Any report that
omits this is misleading, so the ceiling row is mandatory in every table.

### Metrics

**`meaning`** — ordinal 0–4:

| Metric | Why |
|---|---|
| QWK | Primary. Penalises distant disagreement more than adjacent |
| Exact accuracy | Interpretable baseline |
| ±1 accuracy | Adjacent-band tolerance, matches how the app consumes it |
| MAE | Magnitude of drift |
| Confusion matrix | Reveals systematic bias (e.g. student never emits 1) |

**`correction`** — edit-level set comparison. An edit matches when span, replacement,
and tag all agree:

| Metric | Why |
|---|---|
| P/R/F1 exact | Full match including tag |
| P/R/F1 span-only | Isolates detection from classification |
| Tag confusion matrix | Which tags the student conflates |
| Weight-bucket confusion | **The one that matters** — conflation *within* a weight bucket does not move the band; across buckets it does |

**Derived bands** — `grammar` / `naturalness` computed from student `correction` vs
from teacher `correction`, using the same calibrated `band_config.json`. This is the
end-to-end number the product actually consumes: it composes edit errors with the
band function.

**Format validity** — parse rate, tag-set violations, strip-equality failures,
`meaning` range violations. Any non-zero rate here needs a serving retry policy
(Phase 9).

**Operational** — p50/p95 latency and peak VRAM at batch 1, for the 7B and 1.5B
ablations.

### Ceiling: teacher self-consistency

Re-grade a ~300-row sample of the test set with the teacher at temperature 0, then
compare to the original labels:

- QWK(teacher, teacher') → ceiling for the student's `meaning` QWK
- F1(teacher, teacher') → ceiling for `correction` F1

A student at 0.85 against a teacher that scores 0.87 with itself is near-perfect
distillation. Without this row, 0.85 looks mediocre. This measurement is cheap and
changes the interpretation of every other number.

### Slices

Aggregate metrics hide the failure that matters. Report per:

- true `meaning` band (is the middle range weak?)
- POS
- profile CEFR
- sentence length bucket
- edit-count bucket (0, 1, 2–3, 4+)

Grouped bootstrap CIs by target word — not by row, since rows share words.

## Related Code Files

- Create: `lexi_research/eval/__init__.py`
- Create: `lexi_research/eval/predict.py` (batch inference over test set)
- Create: `lexi_research/eval/metrics.py` (QWK, edit-level P/R/F1, bootstrap)
- Create: `lexi_research/eval/slices.py`
- Create: `lexi_research/eval/ceiling.py` (teacher re-grade)
- Create: `lexi_research/eval/report.py` (markdown + parquet emit)
- Create: `lexi_research/eval/cli.py`
- Create: `tests/eval/test_metrics.py`
- Output: `reports/eval-results.parquet`, `reports/eval-summary.json`,
  `reports/eval-report.md`, `reports/errors.parquet`

## Implementation Steps

1. `predict.py`: load base + adapter, greedy decode over test set, persist raw
   generations (never discard raw output — error analysis needs it).
2. `metrics.py`: QWK, exact/±1/MAE, edit-set matching at exact and span-only levels,
   grouped bootstrap. Unit-test each against hand-computed fixtures.
3. Derived-band comparison using the calibrated config.
4. `ceiling.py`: teacher re-grade on ~300 rows; cache to avoid repeat spend.
5. `slices.py`: per-slice aggregation.
6. `report.py`: markdown tables with the ceiling row adjacent to every student row.
7. `errors.parquet`: worst disagreements, sorted, for manual reading.
8. Run for 7B, 1.5B, and `meaning`-only. Log all to W&B.

## Success Criteria

- [ ] `lexi-research eval --adapter ... --split test` produces all four artifacts
- [ ] Every metric table includes the teacher self-consistency ceiling
- [ ] Metric functions unit-tested against hand-computed fixtures
- [ ] Slices reported; no metric presented only as a single aggregate
- [ ] Bootstrap CIs grouped by target word
- [ ] Weight-bucket tag confusion reported (the band-relevant view)
- [ ] Latency p50/p95 and peak VRAM for both model sizes
- [ ] `errors.parquet` readable and sorted by disagreement magnitude
- [ ] Report states plainly: fidelity, not accuracy; no human gold

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Reporting fidelity as accuracy | high | Mandatory ceiling row; explicit statement in report and model card |
| Test set contaminated by iteration | high | Sealed until Phase 7 config is locked; single-use |
| Edit matching too strict (off-by-one span) | medium | Span-only variant alongside exact; normalise whitespace before matching |
| Aggregate hides middle-band weakness | medium | Per-band slices mandatory |
| Teacher re-grade cost | low | 300 rows, cached |
