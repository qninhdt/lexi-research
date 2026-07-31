---
title: "Sentence grader distillation pipeline"
description: "Distil a frontier grader into a QLoRA Qwen2.5-7B student that scores learner sentences (meaning band + inline correction + feedback), with DVC/W&B lineage, Colab training, and an OpenAI-compatible serving shim."
status: in_progress
priority: P1
effort: "3-5 weeks"
tags: [ml, distillation, qlora, nlp, mlops]
created: 2026-07-29
---

# Sentence grader distillation pipeline

## Overview

Build `lexi-research`: a standalone repo that distils a frontier LLM grader into a
small local student model. The student grades a learner-written English sentence
against **one specific dictionary sense** and returns an inline correction, a
meaning band, and one line of feedback.

Design doc: [`docs/grader-distillation-design.md`](../../docs/grader-distillation-design.md)
(authoritative for I/O, taxonomy, band formula).

**Goal is learning + resume.** Not cost reduction, not latency. Stated plainly so
scope decisions stay honest.

### Why this task

`lexi-ai` grades free-text answers for question type `use_in_sentence` through
`grade_rubric` (`lexi_ai/questions/scoring.py`). That is the only LLM call in the
product whose input is **unbounded and uncacheable** — cost scales with usage, not
with vocabulary size. Every other LLM call site (`senses_generation`, `wsd`,
`contextual_mcq`, `example_augment`) is bounded by a ~113K word vocabulary and
permanently cached, so distilling those is dominated by just generating them once.

### Boundary

- **In scope:** dataset generation → QLoRA training → evaluation → serving shim → model card.
- **Out of scope:** any change to `lexi-ai` or `pycil`. Both are read-only here.
  The only read is `lexi-ai/data` (SQLite, Cambridge reference, pinned by SHA-256).
- **Out of scope:** pass/fail verdict. The model **measures**; the app **decides**
  (threshold lives in `pycil`, tunable without retraining).
- **Task `define` deferred.** It has no call site in `pycil` — building it now
  spends ~30% of the generation budget on an unused feature. Schema admits it later.

### The two-call architecture (core decision)

```
CALL 1 — diversifier          CALL 2 — teacher
  knows the spec                sees only {target, sense, text}
  writes learner-like text      EXACT inference prompt
  spec = metadata               output = GROUND TRUTH
```

Call 1's spec (learner profile, target meaning band, error recipe) exists **only to
diversify text**. It is never a label.

Call 2 runs the identical prompt the student will run at inference. That is what
makes this distillation in the strict sense: the student imitates a *function*, not
a dataset that a teacher happened to emit. It also removes the failure mode of
single-call self-labelling, where the label is a copy of the instruction rather
than a reading of the text — a defect that is invisible after the fact because the
output looks the same either way.

### Bands are computed, not generated

The model emits `correction`, `meaning`, `feedback`. Code derives `grammar` and
`naturalness` from the correction's error tags:

```
penalty(group) = Σ weight(tag ∈ group) / √(word_count)
band           = threshold(penalty)
```

Three consequences: identical error sets always score identically (a model emitting
bands disagrees with itself); the anchor is an explicit weight table instead of a
vague rubric; thresholds are tunable **without retraining**, whereas model-emitted
labels are frozen into weights.

## Goals

| # | Goal | Priority |
|---|------|----------|
| 1 | Deterministic, resumable dataset pipeline with full lineage (DVC + R2) | P1 |
| 2 | Format core: `[A>B:tag]` parser + band calculator, exhaustively unit-tested | P1 |
| 3 | Prompt parity — one prompt file used by call 2, training, eval, and serving | P1 |
| 4 | ~20K validated training rows, coverage-measured across the band grid | P1 |
| 5 | QLoRA Qwen2.5-7B trained on Colab Pro, tracked in W&B | P1 |
| 6 | Eval harness reporting fidelity vs teacher, not accuracy vs truth | P1 |
| 7 | OpenAI-compatible serving shim that computes bands server-side | P2 |
| 8 | Model card documenting the no-gold limitation honestly | P2 |
| 9 | Ablations: 7B vs 1.5B, full vs `meaning`-only | P3 |

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Phase 1: Repo scaffold and format core](./phase-01-start.md) | Complete |
| 2 | [Phase 2: Teacher client and prompt contract](./phase-02-teacher-client-and-prompt-contract.md) | Complete |
| 3 | [Phase 3: Sense export and learner profiles](./phase-03-sense-export-and-learner-profiles.md) | Complete |
| 4 | [Phase 4: Generation pipeline call 1 and call 2](./phase-04-generation-pipeline-call-1-and-call-2.md) | In progress — external pilot pending |
| 5 | [Phase 5: Validation balance and split](./phase-05-validation-balance-and-split.md) | In progress |
| 6 | [Phase 6: Band calibration](./phase-06-band-calibration.md) | In progress — reference collection pending |
| 7 | [Phase 7: QLoRA training on Colab](./phase-07-qlora-training-on-colab.md) | Pending |
| 8 | [Phase 8: Evaluation harness](./phase-08-evaluation-harness.md) | Pending |
| 9 | [Phase 9: Serving shim](./phase-09-serving-shim.md) | Pending |
| 10 | [Phase 10: Model card and writeup](./phase-10-model-card-and-writeup.md) | In progress — runtime results pending |

Critical path: 1 → 2 → 3 → 4 → 5 → 7 → 8. Phase 6 gates 8 (bands must be
calibrated before reporting band metrics). Phase 9 depends on 7. Phase 10 last.

## Architecture

```
lexi-ai/data (SQLite, sha256 pinned, READ-ONLY)   profiles.yaml (git)
        │                                                │
        ▼ export-senses                                  │
senses_pool.parquet ─────────── sample-batches ◄─────────┘
                                      │
                                      ▼  batch_specs.parquet
                          ┌───────────────────────────┐
                          │ CALL 1  diversifier       │  K texts / sense
                          │ knows spec                │
                          └───────────┬───────────────┘
                                      ▼  raw_texts.parquet
                          ┌───────────────────────────┐
                          │ CALL 2  teacher           │  = inference prompt
                          │ blind to spec             │
                          └───────────┬───────────────┘
                                      ▼  raw_labels.parquet   ← ground truth
                                      ▼  validate (6 checks)
                       clean.parquet ─┴─ rejects.parquet
                                      ▼  band calc (code)
                                      ▼  balance + split by TARGET WORD
                          train / val / test .parquet
                            │                    │
                            ▼                    ▼
                    calibrate bands        QLoRA train
                  band_config.json        adapter → W&B
                            └──────┬─────────────┘
                                   ▼
                            eval → reports/
                                   ▼
                   serving shim (bands computed here) → vLLM + LoRA
```

### Repo layout

```
lexi-research/
├── lexi_research/
│   ├── format/          parser, band calculator, validators   ← pure, no I/O
│   ├── teacher/         OpenAI-compatible client, prompts/
│   ├── data/            export, sample, generate, validate, split
│   ├── train/           QLoRA entrypoint
│   └── eval/            metrics, reports
├── serve/               FastAPI shim
├── prompts/             grader_system.md  ← THE prompt (call 2 + train + serve)
├── profiles.yaml        learner profiles
├── params.yaml          DVC params
├── band_config.json     weights + thresholds (ships WITH the model)
├── dvc.yaml
├── notebooks/           Colab launcher only — never in the data path
└── tests/
```

### MLOps split

| Tool | Owns |
|---|---|
| Git | code, prompts, `params.yaml`, `profiles.yaml`, `band_config.json`, manifests, reports |
| DVC + Cloudflare R2 | parquet artifacts, lineage. Cambridge-derived text stays **private** (redistribution rights unverified) |
| W&B | runs, checkpoints, confusion matrices, error tables |
| HF Hub | accepted model only |
| CI | format-core tests on synthetic fixtures. No training, no generation |

Lineage identity for any result:
`git_sha + source_db_sha256 + dvc_hash + split_version + params_hash + seed + base_model_revision + teacher_model + prompt_hash`

## Measured facts (verified, not assumed)

| Fact | Value |
|---|---|
| Cambridge senses total | 202,607 |
| Senses with definition + non-empty POS | 202,187 |
| Senses with definition + usable **lexical** POS | 199,863 |
| — minus exact duplicate senses → **the pool** | **183,823** |
| Distinct headwords in pool (split groups) | 99,212 |
| Core examples (`is_extra=0`) | 300,922 |
| Senses with CEFR label | 16,817 |
| `entries.pos` is dirty | `''`, `'V'`, `'adj'`, plus `abbreviation`/`symbol`/`prefix`/`suffix`/`combining form` |

All counts above are at **sense** level — the unit the dataset samples. Entry-level
counts differ substantially (e.g. noun: 100,859 senses vs 74,258 entries) and must
not be substituted.

The pool is **183,823**, not the 199,863 that passes the POS filter: the source
repeats 16,186 senses verbatim across dictionary editions (`appreciate` [verb]
"to increase in value" appears 5×). Content-addressed `sense_uid` collapses them,
which is the intended behaviour — paying a teacher five times for one sense buys
five identical rows. Measured by running the export, not estimated.

The POS defect is load-bearing for Phase 3: export must normalise and filter to
lexical POS only, or the generator receives nonsense targets.

## Success Criteria

- [ ] `uv run pytest` green; mypy strict clean on `lexi_research/`
- [ ] `dvc repro` rebuilds every artifact deterministically from the pinned source
- [ ] Format core: parser round-trips, band calculator unit-tested at boundaries
- [ ] Prompt parity enforced by a test — call 2, training, and serving load the same file
- [ ] ~20K validated rows; coverage report shows no empty band cell
- [ ] Generation resumable: kill mid-run, restart, no duplicate spend
- [ ] QLoRA run completes on Colab Pro; W&B has curves + lineage + checkpoint
- [ ] Eval reports QWK/±1 vs teacher, correction P/R/F1, format validity, teacher self-consistency
- [ ] `band_config.json` calibrated and versioned alongside the adapter
- [ ] Serving shim answers OpenAI-compatible requests with all five fields
- [ ] Model card states the no-gold limitation and the teacher ceiling explicitly

## Risks

| Risk | Level | Mitigation |
|---|---|---|
| **No gold labels** — every metric anchors to the teacher; student inherits its bias | high | Deferred by decision. Model card must say so. Claim *fidelity*, never accuracy |
| **Train and test both teacher-generated** — real learner distribution unknown | high | Deferred. Largest validity threat in the design. Gold sample is future work |
| Batch correlation — K samples per call collapse into one template | med | **K = 6 (fixed)**; spread distinct grid cells inside a batch; measure distinct-n and reject homogeneous batches |
| Batch size changes call-2 judgement (K=6 vs K=1) | med | Phase 2 parity check on ~40 samples; reduce K or unbatch call 2 if it drifts |
| Band weights/thresholds uncalibrated | med | Phase 6. Lives in code — fixable without retraining |
| `naturalness` is not truly span-local | med | `unnat` covers wide spans; documented as an approximation, not an equivalence |
| Teacher self-inconsistency makes distillation meaningless | med | Measure before bulk generation. Fail → fix rubric, do not train |
| Model silently edits untouched text | med | Validator #3: strip markup must equal input exactly |
| No unseen words — every target is in Cambridge | low | Production `generate_fenced` mints new words. Documented limitation |
| Output schema ≠ production `Judgment{correct,score,feedback}` | low | v1 does not deploy. Later: a small PR in `lexi-ai` (prompt + schema + mapping) |
| Colab session loss mid-training | low | Checkpoint to Drive + W&B resume |

## Open Questions

1. `threshold(penalty) → band` cut points — no values yet, Phase 6 produces them.
2. Rows per `(target, sense)` — diversity vs cost. Phase 4 pilot informs it.
3. When/how many gold samples, and who scores them.
4. Ship the 1.5B ablation for a future latency story, or skip it?
5. R2 bucket + credentials — needs to exist before Phase 3 `dvc push`.

## Validation Log

### Session 1 — 2026-07-29

**Trigger:** `/ak:plan validate` after initial plan authoring.
**Tier:** Full (10 phases → all 4 verification roles).
**Questions asked:** 4.

#### Verification Results

- Claims checked: 24 (file paths, symbols, numeric source facts, cross-file contract consistency)
- **Verified: 20 | Failed: 4 | Unverified: 0**

Verified: `lexi_ai/llm.py`, `lexi_ai/questions/scoring.py` exist; symbols
`grade_rubric` (scoring.py:74), `guarded_messages` (llm.py:67), `StructuredLLM`
(llm.py:83), `ainvoke_structured` (llm.py:194) all resolve; 146,144 entries;
202,607 senses; 16,817 CEFR-labelled; 300,922 core examples; 16 tags consistent
across all files; 6 validation checks consistent across all files.

Failed (all in Phase 3 — **root cause: entry-level counts mislabelled as
sense-level**; sense is the unit the dataset actually samples):

| Claim | Actual | Location |
|---|---|---|
| pool = 196,748 senses | **199,863** | `plan.md:184`, `phase-03:48`, `phase-03:163` |
| def + non-empty pos = 199,842 | **202,187** | `phase-03:47` |
| POS dist: noun 74,258 / adj 24,227 / verb 19,384 / idiom 8,367 / adv 5,660 / phrasal 5,530 / coll 3,406 / phrase 2,824 | noun **100,859** / adj **33,223** / verb **34,424** / idiom **8,869** / adv **7,762** / phrasal **6,711** / coll **4,011** / phrase **3,031** | `phase-03:58-59` |
| 7,785 entries carry `{sb}/{sth}` placeholders | **897** — brace tokens do not exist in Cambridge; only bare `sb`/`sth` | `phase-03:71` |

Additional fact discovered and added: pool contains **99,556 distinct
headwords**, which bounds the group count available to Phase 5's split.

Contract inconsistency found: `K` was stated as `6..8` (Phase 2), `6` (Phase 4),
and `≤ 8` (plan.md).

#### Questions & Answers

1. **[Assumptions]** Verification found 4 numeric errors in Phase 3 caused by
   reporting entry-level counts as sense-level. How should this be handled?
   - Options: Fix all + label the measurement level (Recommended) | Fix numbers only | Re-measure everything from scratch
   - **Answer:** Fix all + label the measurement level
   - **Rationale:** Sense is the sampling unit; an entry-level number silently
     understates the pool by ~3K and misstates every POS proportion. Labelling
     the level prevents the same conflation recurring.

2. **[Architecture]** `K` (samples per call) was inconsistent across files.
   Larger `K` is cheaper but raises intra-batch correlation.
   - Options: Fix K=6 everywhere (Recommended) | Fix K=8 | Leave as a tunable range
   - **Answer:** Fix K=6 everywhere
   - **Rationale:** A single hard-coded value makes the Phase 2 parity test
     meaningful — a range would leave the tested configuration ambiguous.

3. **[Scope]** Only 897 entries carry bare `sb`/`sth` placeholders; 47,704
   entries are multiword. At sense level the pool holds idiom 8,869 / phrasal
   6,711 / collocation 4,011 / phrase 3,031, which grade differently from single
   words.
   - Options: Keep all, flag `is_multiword` + `is_placeholder` (Recommended) | Exclude multiword from v1 | Keep only single words + phrasal verbs
   - **Answer:** Keep all, flag both
   - **Rationale:** Multiword targets are legitimate learning targets in Pycil.
     Flagging defers the sampling-weight decision to Phase 4 where measured tag
     distributions exist, rather than guessing now.

4. **[Risks]** The plan had no hard gate after the Phase 4 pilot. Teacher
   self-inconsistency or a collapsed `meaning` distribution would only surface
   at Phase 8, after training.
   - Options: Hard gate — stop if self-consistency low or middle bands missing (Recommended) | Soft warning only | Gate on format validity alone
   - **Answer:** Hard gate
   - **Rationale:** Teacher self-consistency is the ceiling on every downstream
     metric. Training on data that failed it produces a model whose errors cannot
     be distinguished from teacher noise. Cheapest possible place to fail.

#### Confirmed Decisions

- Phase 3 numbers corrected to sense level, with the measurement level named
  explicitly in the table and an entry-vs-sense contrast row added.
- `K = 6` fixed project-wide; Phase 2 parity test is `K=6` vs `K=1`.
- `senses_pool.parquet` gains `is_placeholder` alongside `is_multiword`.
- Phase 4 pilot gate gains two rows: **teacher self-consistency** (re-grade 50
  rows blind, QWK ≥ 0.7 on `meaning`, edit-F1 ≥ 0.6 on `correction`) and
  **middle-band presence** (`meaning` ∈ {1,2,3} ≥ 40%).
- Phase 7 gains a hard entry gate: training is blocked unless the Phase 4 pilot
  gate passed and Phase 6 emitted `band_config.json`.
- Phase 7 `dependencies` widened from `[5]` to `[4, 5, 6]`.

#### Action Items

- [x] Correct Phase 3 source-audit table + POS distribution + placeholder claim
- [x] Correct `plan.md` source-facts table
- [x] Add `is_placeholder` to the parquet schema and success criteria
- [x] Unify `K = 6` across `plan.md`, Phase 2, Phase 4
- [x] Add self-consistency + middle-band rows to the Phase 4 pilot gate
- [x] Add Phase 7 entry gate and widen its dependencies
- [ ] Create the R2 bucket before Phase 3 `dvc push` (user action)

#### Impact on Phases

- **Phase 3:** numbers corrected; `is_placeholder` added to schema, steps, and
  success criteria; success target now 199,863 rows.
- **Phase 4:** pilot gate expanded to 8 rows; self-consistency measured before
  bulk spend; gate now blocks Phase 7 as well as full generation.
- **Phase 7:** entry gate added; dependencies `[4, 5, 6]`.
- **Phase 2:** `K = 6` fixed; parity test restated as `K=6` vs `K=1`.

### Whole-Plan Consistency Sweep

Swept `plan.md` + all 10 phase files for stale terms after propagation.

- Stale-number sweep: zero remaining occurrences of `196,748`, `199,842`,
  `74,258`, `24,227`, `19,384`, `8,367`, `5,660`, `5,530`, `3,406`, `2,824`,
  `7,785` outside the Validation Log's own before/after record.
- `K` sweep: all sites read `K = 6`; no `6..8` or `≤ 8` remains.
- Superseded-decision sweep: 8 hits for `drop rule`, `logprob`, `ERRANT`,
  `BEA-2019`, `verdict`, `intent`, `corrupt`, `coverage` — all confirmed to be
  **explicit negations** ("there is no drop rule", "the teacher does not expose
  logprobs", "rejected: ERRANT on BEA-2019", "out of scope: verdict"), not stale
  content. Retained deliberately as decision provenance.
- Design-doc link corrected to `docs/grader-distillation-design.md` (the earlier
  link pointed at a filename that was never written).
- No write references to `lexi-ai` or `pycil` in any phase — both remain
  read-only, as instructed.

**Unresolved contradictions: 0.**

<!-- slug: sentence-grader-distillation-pipeline -->
