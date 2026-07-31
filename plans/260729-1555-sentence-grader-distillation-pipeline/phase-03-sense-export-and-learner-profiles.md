---
phase: 3
title: "Sense export and learner profiles"
status: complete
priority: P1
effort: "2d"
dependencies: [1]
---

# Phase 3: Sense export and learner profiles

## Overview

Export the sense pool from the Cambridge SQLite source into a fingerprinted
Parquet artifact, and author the learner-profile registry that conditions call 1.

This is the only stage that touches `lexi-ai`, and it is **read-only**. The source
file is treated as immutable input identified by SHA-256.

## Requirements

**Functional**

- Fingerprint the source DB (SHA-256) before reading; record in the manifest
- Extract `(target, pos, definition, cefr)` per sense
- Normalize POS; drop non-lexical entry types
- Stable content-derived `sense_uid` (not the SQLite autoincrement id)
- Quarantine rejected rows with a reason rather than dropping silently
- Emit `senses_pool.parquet` + `data-quality.json` + `source-manifest.json`
- Author `profiles.json`: N learner profiles with distinct error signatures

**Non-functional**

- Deterministic: same source + same code → identical artifact hash
- Cambridge-derived text stays **private** (DVC/R2, never Git, never HF)
- Source path from config; no absolute local path baked into tracked files

## Architecture

### Source audit (measured, not assumed)

Verified against `/home/qninh/projects/lexi-ai/data`:

Verified against `/home/qninh/projects/lexi-ai/data`. **Every figure below is a
SENSE-level count unless the row says "entries"** — the sampling unit is the
sense, and an earlier draft of this phase mistakenly quoted entry-level numbers
under sense labels (corrected in Validation Session 1).

| Fact | Level | Value |
|---|---|---|
| senses total | sense | 202,607 |
| senses with non-empty definition | sense | 202,607 (no empty/null found) |
| senses with definition + non-empty pos | sense | 202,187 |
| senses with **usable lexical** pos | sense | 199,863 |
| exact duplicate senses removed by `sense_uid` | sense | 16,186 |
| → **the pool** | sense | **183,823** |
| distinct headwords in the pool | target | **99,212** |
| examples (`is_extra=0`) | example | 300,922 |
| entries | entry | 146,144 |
| multiword headwords | entry | 47,704 |
| headwords carrying bare `sb`/`sth` placeholders | entry | 897 |

`entries.pos` is **dirty** and must be normalized. Observed values include the
canonical set plus `''`, `'V'`, `'adj'`, and non-lexical types.

POS distribution among usable senses (**sense-level**; the entry-level figure is
shown alongside because the two differ by ~1.4× and confusing them is exactly the
error this table now guards against):

| POS | Senses | Entries |
|---|---|---|
| noun | 100,859 | 74,258 |
| verb | 34,424 | 19,384 |
| adjective | 33,223 | 24,227 |
| idiom | 8,869 | 8,367 |
| adverb | 7,762 | 5,660 |
| phrasal verb | 6,711 | 5,530 |
| collocation | 4,011 | 3,406 |
| phrase | 3,031 | 2,824 |
| exclamation | 511 | 436 |
| plural noun | 462 | 424 |

`99,212` distinct headwords over `183,823` pool senses ≈ 1.9 senses/target. This
matters for Phase 5: grouped splitting by `target_norm` has ~99.2K groups, which
is ample for a 70/15/15 grouped split.

The POS table above counts senses **passing the POS filter**, before dedup. The
pool is 16,186 rows smaller because the source repeats senses verbatim across
dictionary editions — same headword, same POS, same definition, up to 5 copies
(`appreciate` [verb] "to increase in value"). `sense_uid` is content-addressed,
so the copies collapse to one row and the rest land in quarantine under
`duplicate_uid`. That is the intended behaviour: five identical rows cost five
teacher calls and add no signal. The per-POS effect is uneven — `verb` loses the
most (34,424 → 21,800) because verb entries are the most duplicated across
editions.

Normalization map: `adj`→`adjective`, `V`→`verb`, `plural noun`→`noun`,
`auxiliary verb`/`modal verb`→`verb`, `ordinal number`→`number`.

**Excluded** (not gradable as "use this word in a sentence"): `''`, `suffix`,
`prefix`, `combining form`, `abbreviation`, `written abbreviation`, `symbol`,
`number`, `ordinal number`, `modifier`, `predeterminer`, `indefinite article`.

**Kept but flagged** `is_multiword`: the four multiword POS values contribute
`idiom` (8,781 exported), `phrasal verb` (6,623), `collocation` (4,011), `phrase`
(3,029) — 22,444 senses. The exported flag is **52,533 senses (28.6%)** rather
than ~12%, because the flag is `multiword POS OR the headword contains a space`:
a large number of `noun`-POS headwords are themselves multiword (`credit card`,
`air traffic control`). That is the correct reading of the flag — the learner has
to produce the whole unit either way — but the two numbers measure different
things and must not be swapped.

These are legitimate learning targets and grade differently: the learner must use
the whole unit, so `part` (particle) errors carry real weight and a "correct word,
wrong particle" answer is a distinct failure mode from single-word targets.

**Kept but flagged** `is_placeholder`: 897 entries carry **bare** `sb`/`sth`
placeholders (e.g. `put sb down`). Two corrections from Validation Session 1:

- The count is **897**, not 7,785 (an earlier draft was off by ~9×).
- The placeholders are **bare tokens** (`sb`, `sth`), not brace tokens. `{sb}` and
  `{sth}` do **not** appear anywhere in the Cambridge DB — brace form is
  `lexi-ai`'s own canonical representation applied downstream of this source.
  Export must match on bare tokens with word boundaries; a brace-token matcher
  silently returns zero rows.

Both flags are carried into `senses_pool.parquet`, and Phase 4 decides sampling
weight. Reported separately in eval (Phase 8) so multiword performance is never
hidden inside an aggregate.

### `senses_pool.parquet`

```
sense_uid          sha256(target_norm|pos|definition)[:16]
source_sense_id    provenance only, never a join key downstream
target             headword
target_norm        normalized for grouping/splitting
pos                normalized
definition
cefr               A1..C2 or null
is_multiword       bool    (idiom / phrasal verb / collocation / phrase)
is_placeholder     bool    (bare `sb` / `sth` in headword)
source_db_sha256
```

`target_norm` is the **split group key** for Phase 5. Splitting by row leaks:
one target yields many rows.

The pool holds **99,212 distinct `target_norm` groups** (measured on the exported
artifact). That is the real group count available to Phase 5 — ample for a grouped
split, and the number to re-check if the pool filter ever tightens.

### Quarantine

`quarantine.parquet` with reasons: `empty_definition`, `excluded_pos`,
`unmappable_pos`, `duplicate_uid`, `definition_too_short`. Counts reported in
`data-quality.json`. Nothing dropped silently.

### Learner profiles

`profiles.json` — a versioned Git artifact, hashed into the manifest.

Each profile is a plausible learner whose **error signature is distinct enough to
be observable**. Not cosmetic variants.

```json
{
  "id": "vi-b1-articles",
  "l1": "Vietnamese",
  "cefr": "B1",
  "error_bias": ["art", "agr", "tense"],
  "length": "short",
  "traits": "omits articles; overuses present simple; word-for-word phrasing from L1"
}
```

Coverage across 4 L1 families with genuinely different transfer errors:

| L1 | Characteristic errors | Tags |
|---|---|---|
| Vietnamese | no articles, no inflection, tense via adverbs | `art` `agr` `tense` `num` |
| Japanese | articles, plural marking, preposition choice | `art` `num` `prep` |
| Spanish | gender/agreement transfer, word order, false friends | `agr` `order` `word` |
| Arabic | copula omission, definiteness, VSO order | `agr` `art` `order` |
| (none — advanced) | collocation and register only | `coll` `unnat` |

× CEFR {A2, B1, B2, C1} × length {short, medium} → select **~16 profiles**,
including 2–3 near-native profiles whose only errors are `coll`/`unnat`. Without
those, the `naturalness` band has no high-quality-but-unnatural examples.

`error_bias` conditions call 1's prompt. It is **not** a label — call 2 decides
what errors actually exist.

## Related Code Files

- Create: `lexi_research/data/export.py`
- Create: `lexi_research/data/pos_normalize.py`
- Create: `lexi_research/data/profiles.py` (loader + validation)
- Create: `lexi_research/data/profiles.json`
- Create: `tests/data/{test_export,test_pos_normalize,test_profiles}.py`
- Create: `tests/fixtures/mini_cambridge.sqlite` (synthetic, ~20 senses, CI-safe)
- Read-only input: `/home/qninh/projects/lexi-ai/data`

## Implementation Steps

1. `pos_normalize.py`: mapping table + `normalize_pos()` + `is_lexical()` +
   `is_multiword()`. Pure functions, exhaustively tested.
2. `export.py`: fingerprint source; stream senses joined to entries; normalize;
   quarantine; compute `sense_uid`; write Parquet + `data-quality.json` +
   `source-manifest.json`.
3. Build the synthetic fixture DB mirroring the real schema — including the dirty
   POS values — so CI never needs the private source.
4. Author `profiles.json` (~16 profiles) with a JSON Schema; validate on load.
5. `profiles.py`: load, validate, deterministic sampling by seed.
6. Tests: POS normalization covers every observed value; export determinism (two
   runs → identical hash); quarantine counts; profile schema validity; profile
   `error_bias` tags all ∈ the 16-tag taxonomy.
7. DVC stage `export`: deps = source fingerprint + code + params; outs =
   `senses_pool.parquet`, `quarantine.parquet`, reports.
8. `dvc push` to R2.

## Success Criteria

- [x] `senses_pool.parquet` = **183,823 rows** over **99,212 distinct `target_norm`
      groups** (measured). 199,863 senses carry a usable lexical POS; 16,186 of
      them are exact `(target_norm, pos, definition)` duplicates and are
      quarantined as `duplicate_uid`
- [ ] Every observed `entries.pos` value either maps or is explicitly excluded —
      zero `unmappable_pos` in quarantine
- [ ] Two consecutive runs produce byte-identical artifact hashes
- [ ] `data-quality.json` reports POS/CEFR/multiword distributions and all
      quarantine reasons with counts
- [ ] ~16 profiles validate; `error_bias` tags all within the taxonomy; ≥2
      near-native profiles present
- [ ] CI passes using only the synthetic fixture — no private data
- [ ] Artifacts in R2; no Cambridge text in Git

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Dirty POS silently drops senses | Explicit map + `unmappable_pos` quarantine gate at zero |
| Multiword targets grade differently, distorting bands | `is_multiword` flag; sampling weight decided in Phase 4 with measured data |
| Cambridge licence forbids redistribution | Private R2; Git holds only hashes/counts; never published to HF |
| Profiles are cosmetic → no real diversity gain | Error signatures drawn from distinct L1 families; Phase 4 measures tag distribution per profile |
| `sense_uid` collision | sha256 over normalized content; collisions quarantined |
| Source DB changes under us | SHA-256 pinned; mismatch fails the stage loudly |
