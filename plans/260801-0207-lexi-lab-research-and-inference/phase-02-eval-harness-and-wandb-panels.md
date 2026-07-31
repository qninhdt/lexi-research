---
phase: 2
title: "Eval harness and W&B panels"
status: done
priority: P1
size: M
dependencies: [1]
---

# Phase 2: Eval harness and W&B panels

# This phase gates everything after it

Phase 4 will very likely produce "RL does not beat SFT". That is a publishable
result *only* if the harness measuring it was trusted before the RL code existed.
Built afterwards, a null result is indistinguishable from a measurement bug. So
this phase ran before any real training, and its own correctness was established
against synthetic inputs with known answers — every metric test asserts a value
computed on paper, never a range.

## Overview

`lexi_research/eval/metrics.py` had QWK and edit-F1, about a fifth of what the
design calls for. This phase adds the rest: correction metrics that separate a
wrong span from a wrong tag, ECE with a reliability diagram, format and taxonomy
metrics, a report that carries its own lineage and ceiling, the teacher-as-judge
path for `feedback`, and W&B panels defined in code.

## Files

**Created**

- `lexi_research/eval/correction.py` — span+tag P/R/F1, span-only F1, tag confusion, cross-tier confusion rate
- `lexi_research/eval/calibration.py` — ECE over equal-mass bins, reliability diagram
- `lexi_research/eval/harness.py` — predictions I/O, ceiling, every metric, assembly
- `lexi_research/eval/predict.py` — the one step that needs a GPU
- `lexi_research/eval/judge.py` — teacher-as-judge, both orders, discards counted
- `lexi_research/eval/report.py` — schema, reliability tags, markdown for the model card
- `lexi_research/tracking/panels.py` — panel and table definitions in code
- `ops/fixtures/build_eval_fixture.py`, `eval_predictions.jsonl`, `eval_ceiling.json`
- `tests/eval/test_correction.py`, `test_reliability.py`, `test_harness.py`,
  `test_report.py`, `test_judge.py`

**Modified**

- `lexi_research/eval/__init__.py` — the public surface
- `lexi_research/cli/__init__.py` — `lexi eval predict` and `lexi eval score`
- `dvc.yaml` — `predict` and `eval` stages
- `params.yaml` — `eval.max_retries`, `calibration_bins`, `judge_sample`
- `ops/Makefile` — `make fixture` builds both fixtures

## What the harness refuses to do

- **Report bands from an uncalibrated config.** The thresholds shipped in
  `band_config.json` are design guesses and the file says `calibrated: false`. A
  grade derived from a guess is a guess, so scoring raises until `lexi data
  calibrate` has run.
- **Report without a ceiling.** Teacher self-consistency is the highest score the
  data can support. A student at 0.72 QWK against a teacher scoring 0.74 against
  itself has nearly saturated the available signal; the same number against a
  ceiling of 0.95 has not. Both headline metrics carry their fraction of ceiling.
- **Print a weak metric as if it were strong.** `feedback` has no verifiable
  ground truth. chrF and the judge win-rate are tagged `reliability: "weak"` in
  the JSON, and the markdown renderer prints the tag on the same line as the
  number.

## Metric decisions

| Metric | Decision |
|---|---|
| span+tag vs span-only F1 | Both, plus the tag error rate between them. A right span with the wrong tag is a different failure from a wrong span, and only the second means the model misread the sentence |
| cross-tier confusion | Counted only when the two tags carry *different* weights. Equal weight cannot move a band — the property the taxonomy asserts for its confusable pairs — so a `word`↔`coll` swap is harmless by construction |
| precision/recall | Micro-averaged over edits, not macro over rows. A hundred clean rows each scoring a free 1.0 would otherwise drown out the rows that carry edits |
| ECE | Equal-mass bins. The band distribution is skewed by design, so equal-width bins leave several nearly empty and ECE becomes noise reported to three decimals |
| per-band breakdown | Every band present, including the empty ones. A band with no rows is a fact about the split, not a key to omit |
| judge | Both orders of every pair; contradictions discarded and the discard rate reported. A model that prefers whichever came second scores 100% on a single-order test |
| `other` rate | Reported next to the teacher's own. A student reproducing the teacher's taxonomy gaps is distilling faithfully |

## The fixture

Eight rows, one per case: an exact hit, a right span with a same-tier tag, a right
span with a cross-tier tag, a clean sentence, a missed edit, a spurious edit, a
row both sides judged beyond correction, and a band off by one. Six gold edits,
six predicted, three matching on span and tag, five on span alone — so the tests
assert 0.5, 5/6 and 0.4 rather than "greater than zero".

## Tests

| Test | Asserts |
|---|---|
| `test_correction.py::test_exact_f1_values` | hand-computed P/R/F1 on known overlaps |
| `test_correction.py::test_span_only_vs_span_tag` | right span, wrong tag: 1.0 span-only, 0.0 span+tag |
| `test_correction.py::test_cross_tier_confusion` | `tense`→`word` (both weight 2) excluded; `tense`→`sp` (2 vs 1) counted |
| `test_correction.py::test_confusable_pairs_are_never_cross_tier` | the taxonomy's own promise, measured through this metric |
| `test_correction.py::test_precision_is_micro_averaged` | clean rows cannot inflate the score |
| `test_reliability.py::test_known_ece` | analytic ECE reproduced to 1e-9 |
| `test_reliability.py::test_equal_mass_bins` | equal counts under a distribution equal-width bins would ruin |
| `test_harness.py::test_refuses_uncalibrated_bands` | raises while `calibrated: false` |
| `test_harness.py::test_requires_ceiling` | raises when the ceiling artifact is absent |
| `test_harness.py::test_calibration_is_skipped_and_said_so_when_confidence_is_absent` | a missing ECE is a note, not a silent omission |
| `test_report.py::test_weak_metrics_tagged` | chrF and judge win-rate carry `reliability: weak` |
| `test_report.py::test_report_is_self_contained` | the JSON alone interprets every number |
| `test_judge.py::test_a_judge_that_always_says_a_is_discarded_entirely` | position bias is discarded, not reported as a win |

## Acceptance

- [x] `lexi eval score` on the fixture reproduces every hand-computed value.
- [x] The report JSON validates against its schema and is interpretable without W&B.
- [x] Reporting bands with `calibrated: false` fails loudly.
- [x] Generation and scoring are separate commands; scoring never loads a model.
- [x] `uv run pytest` green — 596.
- [ ] A W&B run rendering all five panel groups — needs a key. Panels are defined
      in `panels.py` and logged through the same handle every stage uses; the
      first real run is where they are eyeballed once.

## Findings

- **The fixture's QWK exceeds its ceiling** (100.6%). Invented numbers, but it
  makes the point the normalisation exists for: a fraction over 100% means the
  student agrees with the teacher more than the teacher agrees with itself, which
  is a signal about the *ceiling measurement*, not about the student.
- **`feedback.chrf` is 0.52 between two paraphrases that say the same thing.**
  Exactly why it is tagged weak — it measures surface overlap, and two good
  answers can share almost none.

## Risks

| Risk | Handling |
|---|---|
| Metrics look right but are subtly wrong | Every test asserts hand-computed exact values; none assert only a range |
| The judge is expensive and slow | Sampled, size is a config value and lands in the report; both orders double the cost and that is stated |
| Panels defined in code diverge from what the UI shows | One fixture run per panel group, inspected by eye once during the first real run |
| The ceiling artifact does not exist yet | `reports/pilot-ceiling.json` is produced by the pilot gate during the first real data build; until then the eval stage cannot run, which is the intended order |
