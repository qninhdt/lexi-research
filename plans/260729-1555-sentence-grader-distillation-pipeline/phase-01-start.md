---
phase: 1
title: "Repo scaffold and format core"
status: complete
priority: P1
effort: "3d"
dependencies: []
---

# Phase 1: Repo scaffold and format core

## Overview

Stand up `lexi-research` as a professional Python project, then implement the
**format core** — the `[A>B:tag]` parser, the band calculator, and the output
validators. These are pure functions with no I/O and no LLM dependency, so they can
be tested exhaustively before a single token is spent.

This phase is first because everything downstream consumes it: call 2's validator,
the training data builder, the eval harness, and the serving shim all parse
corrections and compute bands through this one module.

## Requirements

**Functional**

- Parse `correction` strings into a typed edit list; reject malformed input
- Render an edit list back to a correction string (round-trip property)
- Strip markup to recover the original text
- Compute `grammar` and `naturalness` bands from parsed edits + word count
- Validate a full grader output against all 6 checks from the design doc
- Load `band_config.json` (weights + thresholds) as versioned data, not constants

**Non-functional**

- Zero I/O in `lexi_research/format/` — pure functions only, trivially testable
- mypy strict clean
- Deterministic: no global RNG, no locale/time dependence
- Parser must not raise on adversarial input; it returns a typed failure

## Architecture

### Package layout

```
lexi-research/
├── pyproject.toml            uv, py>=3.10, ruff, mypy strict, pytest
├── lexi_research/
│   ├── __init__.py
│   └── format/
│       ├── __init__.py       public API re-exports
│       ├── tags.py           Tag enum, TagGroup, taxonomy constants
│       ├── parser.py         parse / render / strip
│       ├── bands.py          BandConfig, penalty, band derivation
│       └── validate.py       6-check output validator
├── band_config.json
└── tests/format/
```

### Correction format

```
He [speak>speaks:agr] very [eloquent>eloquently:form].
```

| Operation | Syntax | Example |
|---|---|---|
| Replace | `[A>B:tag]` | `[speak>speaks:agr]` |
| Delete | `[A>:tag]` | `the [the>:art] very` |
| Insert | `[>B:tag]` | `went [>to the:art] store` |

Regex: `\[([^\]>]*)>([^\]:]*):([a-z]+)\]`

Rules:
- A clean sentence is emitted verbatim → **zero token overhead**
- Unparseable sentence → `correction: null` (no attempt to invent edits)
- Escape: a literal `[` in learner text is `\[`
- Both `original` and `replacement` empty is invalid (`[>:tag]`)

### Taxonomy — 16 tags

| Group | Weight | Tags |
|---|---|---|
| Correctness | 1 | `punc` `sp` `art` `num` `poss` |
| Correctness | 2 | `prep` `part` `agr` `tense` `form` `pron` |
| Correctness | 3 | `order` |
| Usage | 2 | `coll` `word` |
| Usage | 3 | `unnat` |
| — | 2 | `other` |

`sp` spelling (absorbs contractions: `dont`) · `agr` subject-verb agreement ·
`tense` tense · `form` word form, absorbs morphology/inflection
(`eloquent`→`eloquently`) · `art` article/determiner · `prep` preposition ·
`part` phrasal particle (`look up`/`look on`) · `num` countability/number ·
`poss` possessive · `pron` pronoun · `order` word order · `punc` punctuation ·
`coll` collocation (`do a decision`) · `word` word choice · `unnat` unnatural
(absorbs register + wordiness) · `other` catch-all

**No tag for wrong meaning.** Wrong meaning does not live in a span and cannot be
fixed by replacement — it lives in the `meaning` band.

**Load-bearing invariant:** tags that are easy to confuse carry **equal weight**.
`word`↔`coll` (2/2), `prep`↔`part` (2/2) — confusing them cannot move a band. This
is what makes deriving bands from noisy tags viable. A test must assert this
property so a future taxonomy edit cannot silently break it.

`other` exists to **measure what the taxonomy is missing**. Without it the teacher
crams misfits into the nearest tag, producing hidden label noise.

### Band derivation

```python
penalty(group) = sum(weight(tag) for tag in edits if group_of(tag) == group) / sqrt(word_count)
grammar        = threshold(penalty(CORRECTNESS))
naturalness    = threshold(penalty(USAGE))
```

`correction is None` → `grammar = 0`, formula skipped.

`√(word_count)` normalisation: 2 errors in a 6-word sentence is worse than 2 errors
in a 30-word sentence.

**The inverted-failure guard matters here.** An unreadable sentence produces no
parseable edits → penalty 0 → band 4, the exact opposite of the truth. `null` is
the escape hatch, and a test must cover it.

### `band_config.json`

```json
{
  "version": 1,
  "calibrated": false,
  "weights": { "punc": 1, "sp": 1, "art": 1, "num": 1, "poss": 1,
               "prep": 2, "part": 2, "agr": 2, "tense": 2, "form": 2,
               "pron": 2, "order": 3, "coll": 2, "word": 2, "unnat": 3,
               "other": 2 },
  "groups": { "correctness": ["punc","sp","art","num","poss","prep","part",
                              "agr","tense","form","pron","order","other"],
              "usage": ["coll","word","unnat"] },
  "thresholds": [0.0, 0.4, 0.9, 1.6]
}
```

`thresholds` are **placeholders** until Phase 6. `calibrated: false` is the honest
marker; Phase 6 flips it. Eval must refuse to report band metrics while it is false.

`other` sits in `correctness` for band purposes — a catch-all error is more likely
grammatical than stylistic.

This file **ships with the model**. A checkpoint without it produces meaningless
bands.

### Validator — 6 checks

1. `correction` parses fully (or is `null`)
2. Every tag ∈ the 16-tag closed set
3. **Strip markup == input `text`, exactly** — mandatory
4. `meaning` ∈ 0..4
5. No empty edit (`[>:tag]`)
6. `feedback` non-empty, single sentence

Check 3 is the one that cannot be dropped. `correction` re-emits the whole sentence,
so without it the model can silently rewrite untouched words and nothing would
notice.

Returns a typed result: `ValidationOk(edits, ...)` or `ValidationError(code, detail)`.
Callers decide policy — drop the row when building data, retry at inference.

## Related Code Files

- Create: `pyproject.toml`, `.gitignore`, `README.md`, `.python-version`
- Create: `lexi_research/format/{__init__,tags,parser,bands,validate}.py`
- Create: `band_config.json`
- Create: `tests/format/{test_parser,test_bands,test_validate,test_taxonomy}.py`
- Create: `.github/workflows/test.yml`
- Read-only reference: `/home/qninh/projects/lexi-ai/pyproject.toml` (tooling conventions to mirror)

## Implementation Steps

1. `git init`; `pyproject.toml` with uv, `requires-python >=3.10`, ruff
   (`line-length = 100`), mypy strict, pytest. Mirror `lexi-ai`'s tool config so
   conventions stay consistent across the two repos.
2. `tags.py`: `Tag` StrEnum (16 members), `TagGroup` enum, `GROUP_OF` mapping.
3. `parser.py`: `Edit` dataclass (`original`, `replacement`, `tag`, `span`),
   `parse_correction(s) -> list[Edit] | ParseError`, `render(text, edits) -> str`,
   `strip_markup(s) -> str`. Handle `\[` escaping.
4. `bands.py`: `BandConfig.from_json`, `penalty(edits, group, word_count)`,
   `derive_bands(...) -> Bands`. Word count on the **stripped** text.
5. `validate.py`: `validate_output(payload, input_text, config) -> ValidationResult`
   running all 6 checks in order, short-circuiting on the first failure.
6. Tests:
   - Parser: replace/delete/insert, multiple edits, clean sentence, `null`,
     escaped `[`, malformed (unclosed, unknown tag, empty edit), adversarial
     (nested brackets, `>` inside text)
   - **Round-trip property**: `parse` → `render` == original for all valid inputs
   - **Strip property**: `strip_markup(correction)` == original text
   - Bands: zero edits → 4; `null` → grammar 0; threshold boundary values;
     length normalisation; group isolation (usage edits must not move grammar)
   - **Taxonomy invariant**: assert confusable pairs share weight
   - Validator: one test per check, each failing in isolation
7. CI: ruff + mypy + pytest on push. No training, no network.

## Success Criteria

- [ ] `uv run pytest` green
- [ ] `uv run mypy --strict lexi_research/` clean
- [ ] `uv run ruff check .` clean
- [ ] Round-trip and strip properties hold across the full valid-input test matrix
- [ ] Parser never raises — malformed input returns a typed error
- [ ] `null` correction yields `grammar = 0`, not 4
- [ ] Taxonomy-invariant test passes (confusable pairs share weight)
- [ ] `band_config.json` loads, and `calibrated: false` is respected downstream
- [ ] CI green on a clean clone

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Insert operation is ambiguous when several positions are valid | Position is explicit in the emitted string (whole sentence re-emitted), so rendering is unambiguous. Test insert at start/middle/end |
| `>` or `[` occurring naturally in learner text | Escape `\[`; `>` is only special inside a bracket group. Add adversarial tests |
| Placeholder thresholds leak into reported results | `calibrated: false` flag; Phase 8 refuses band metrics until Phase 6 flips it |
| Taxonomy churn later breaks the equal-weight invariant | Explicit test asserting the invariant |
| Word count differs between raw and stripped text, skewing penalty | Always count on stripped text; assert in tests |
