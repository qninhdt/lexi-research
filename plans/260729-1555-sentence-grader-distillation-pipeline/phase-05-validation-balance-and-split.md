---
phase: 5
title: "Validation, balance, and split"
status: pending
priority: P1
effort: "1.5d"
dependencies: [4]
---

# Phase 5: Validation, balance, and split

## Overview

Turn `raw_labels` into train/val/test splits: apply the six structural checks,
compute derived bands, balance strata, and split by **target word** so no lemma
crosses splits.

## Requirements

**Functional**

- Six validation checks, each producing a named rejection reason
- Rejected rows preserved in `rejects.parquet` with reason + detail (never deleted)
- Derived columns computed: `grammar`, `naturalness`, `n_edits`, `tags[]`, `n_words`
- Grouped split by `lemma_key`; ratio 80/10/10
- Sentence-hash contamination report across splits
- Balance: cap over-represented strata, report distribution before/after

**Non-functional**

- Deterministic given `(dataset, seed, split_version)` — identical hashes on re-run
- Rejection rate broken down by reason (a pipeline health signal)

## Architecture

### Six checks

| # | Check | Reason code |
|---|---|---|
| 1 | `correction` parses fully as `[A>B:tag]` markup | `parse_error` |
| 2 | Every tag ∈ the 16-tag closed set | `unknown_tag` |
| 3 | **Strip markup == input `text`** | `text_mismatch` |
| 4 | `meaning` ∈ 0..4 | `meaning_range` |
| 5 | No empty edit `[>:tag]` | `empty_edit` |
| 6 | `feedback` non-empty, single sentence | `feedback_shape` |

**Check 3 is the one that matters most.** With re-emit-whole-sentence format, a
model can silently alter untouched parts of the sentence. Stripping markup and
comparing against the input is the only thing that catches it. Without this check,
corrupted rows enter training undetected.

Rejects are kept, not dropped: the reason distribution tells you whether the prompt,
the schema, or the taxonomy is at fault.

### Derived bands

Applies `format/bands.py` (Phase 1) using `band_config.json`. Phase 6 recalibrates
the thresholds; this stage just applies whatever config is current, and the config
hash is recorded so a dataset always states which calibration produced its bands.

### Split by target word

Grouped on `lemma_key`. One sense yields many rows and one lemma yields many
senses, so splitting by row — or even by sense — leaks: the model would see the same
target word in train and test.

Ratio 80/10/10, grouped-stratified to approximate the `meaning` distribution across
splits.

### Contamination report

The same sentence can legitimately appear under different senses. After splitting,
report identical `text` hashes crossing splits and build a **strict test subset**
excluding any sentence seen in train/val. Report standard and strict metrics
separately in Phase 8.

### Balance

Some strata will be over-produced (clean sentences are easy for the teacher to
write). Cap them so the objective is not dominated. Report band and tag
distribution before and after, and never balance by discarding a rare stratum.

## Related Code Files

- Create: `lexi_research/data/validate.py`
- Create: `lexi_research/data/derive.py` (band + tag columns)
- Create: `lexi_research/data/split.py`
- Create: `lexi_research/data/balance.py`
- Create: `lexi_research/data/report.py` (dataset card + quality JSON)
- Create: `tests/data/{test_validate,test_split,test_balance}.py`

## Implementation Steps

1. `validate.py`: six checks, each returning a typed reason; split rows into
   `clean` / `rejects`.
2. `derive.py`: parse `correction` → tags, edit count, word count → `grammar`,
   `naturalness` via `format/bands.py`; record `band_config` hash.
3. `balance.py`: cap over-represented strata by `(meaning, error_bucket)`; emit
   before/after distributions.
4. `split.py`: grouped split on `lemma_key`, grouped-stratified by `meaning`;
   emit `splits.parquet` with `split_version` + `seed`.
5. Contamination: cross-split sentence-hash report + strict test subset.
6. `report.py`: `data-quality.json`, `dataset-manifest.json`, `dataset-card.md`
   (provenance, exclusions, limitations, license uncertainty).
7. Tests: leakage assertion (no `lemma_key` in two splits), determinism (same
   seed → same hashes), each check triggers on a crafted bad row.
8. DVC stages `validate` → `balance` → `split`; freeze `dataset-v1`.

## Success Criteria

- [ ] Six checks implemented; every reason code exercised by a test
- [ ] Rejection rate by reason reported; overall < 10%
- [ ] Zero `lemma_key` overlap across splits (asserted by test)
- [ ] Deterministic rebuild → identical artifact hashes
- [ ] Strict test subset exists; contamination quantified
- [ ] `dataset-card.md` states teacher model, prompt hash, source DB hash, and the
      deferred-validity limitation
- [ ] `dataset-v1` frozen in DVC before any training run

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Silent sentence alteration by teacher | high | Check 3 (strip == input) |
| Lemma leakage inflating eval | high | Grouped split + assertion test |
| Balancing distorts the distribution | medium | Cap only; report before/after; never drop rare strata |
| High rejection rate wastes generation spend | medium | Pilot gate in Phase 4 catches format problems first |
| Band config drift between dataset and model | medium | Config hash recorded in the dataset manifest |
