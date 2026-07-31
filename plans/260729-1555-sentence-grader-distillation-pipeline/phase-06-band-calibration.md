---
phase: 6
title: "Band calibration"
status: pending
priority: P2
effort: "1d"
dependencies: [5]
---

# Phase 6: Band calibration

## Overview

Replace the hand-guessed tag weights and penalty thresholds in `band_config.json`
with values fitted against teacher judgement, and verify the confusability
invariant still holds.

## Requirements

**Functional**

- Fit `threshold(penalty) → band` boundaries from data rather than assumption
- Verify/adjust tag weights against observed teacher behaviour
- Produce a tag confusion matrix from the double-grade sample (Phase 2)
- Emit a calibrated `band_config.json` with a version and provenance record

**Non-functional**

- Calibration is a pure post-processing step: no regeneration, no retraining
- Old and new configs both retained; a dataset states which produced its bands

## Architecture

### Why calibration is separate from training

Bands are computed in code from `correction`. That means thresholds can be refit at
any time without touching the model — the property that motivated the derived-band
design in the first place. This phase exercises that property once, deliberately,
so the shipped defaults are not arbitrary.

### Calibration signal

There is no human gold (deferred). The available signal is the teacher itself:

1. **Reference bands.** On a calibration sample (~300 rows held out of train), ask
   the teacher for a *holistic* `grammar` and `naturalness` band directly, in a
   separate prompt that is **not** the inference prompt. This is used only to fit
   thresholds — never as a training label, and never merged into the dataset.
2. **Fit thresholds** so the formula's output best matches those reference bands
   (maximise QWK).
3. **Weight sanity check.** For each tag, compare the reference band of rows
   containing that tag alone against rows with no errors. A tag whose presence
   barely moves the reference band should not carry weight 3.

This keeps the split clean: model learns `correction`; code converts `correction`
to bands; calibration only tunes the conversion.

### Confusability invariant

The design's load-bearing property: tags the teacher confuses must carry equal
weight, so a labelling mistake cannot move a band.

From the double-grade sample, build a tag confusion matrix. Any pair confused above
a threshold **and** carrying different weights is a defect — fix by equalising
weights or merging the tags. This check runs in CI as a config test, so a future
weight edit cannot silently break the invariant.

### `other` rate

If `other` exceeds 5%, the taxonomy is missing a real category. Inspect what landed
there and decide: add a tag (and regenerate labels), or accept and document. This is
a taxonomy decision, so surface it rather than deciding silently.

## Related Code Files

- Create: `lexi_research/eval/calibrate.py`
- Create: `lexi_research/eval/confusion.py`
- Modify: `config/band_config.json` (calibrated values + version + provenance)
- Create: `tests/eval/test_calibration_invariant.py`
- Create: `reports/calibration-v1.md`

## Implementation Steps

1. Sample ~300 calibration rows (held out of train).
2. Write the holistic band prompt (separate file, clearly marked
   calibration-only, never used at inference).
3. Collect reference bands; fit `penalty → band` thresholds maximising QWK.
4. Per-tag weight sanity check; adjust weights where evidence contradicts the guess.
5. Build the tag confusion matrix from the Phase 2 double-grade sample.
6. Assert the confusability invariant; equalise or merge on violation.
7. Report the `other` rate; escalate if > 5%.
8. Emit calibrated `band_config.json` with version + provenance; write
   `reports/calibration-v1.md`.
9. CI test: invariant holds for the committed config.

## Success Criteria

- [ ] Thresholds fitted, not guessed; QWK vs reference bands reported
- [ ] Tag weights justified by evidence or explicitly marked as unchanged defaults
- [ ] Confusability invariant asserted in CI
- [ ] `other` rate reported; decision recorded if > 5%
- [ ] `band_config.json` carries version + provenance; recorded in the dataset manifest
- [ ] Calibration prompt is separate from and never substituted for the inference prompt

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Calibration prompt leaks into training data | high | Separate file, marked calibration-only, excluded from dataset stages |
| Circular calibration (teacher tunes teacher) | medium | Acknowledged limitation; documented in model card. Human gold would break the circle later |
| Fitted thresholds overfit 300 rows | medium | Report bootstrap CI; prefer round numbers when the interval is wide |
| Future weight edit breaks the invariant | medium | CI config test |
