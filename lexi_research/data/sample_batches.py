"""Sample the generation plan: which senses, which specs, in which batches.

Everything here is metadata for call 1. `meaning_req` and `error_spec` steer what
call 1 writes; call 2 then reads the resulting sentence blind and decides what is
actually true of it. Nothing sampled here is ever a label, and a test asserts the
spec columns never reach a training-label path.

Two properties do real work:

**Weighted toward the middle.** Left unweighted, a teacher asked for "a learner
sentence" writes either a clean one or nonsense, and bands 1-3 — where real
learners live and where grading is hardest — come out empty. The weights in
`params.yaml` bias sampling toward them.

**Distinct cells inside a batch.** K specs in one call share a sense, so the model
will happily write K near-identical sentences. Forcing different grid cells into
the same batch makes variety structural rather than a hope about prompt wording.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexi_research.teacher import DiversifySpec

from .profiles import Profile, ProfileRegistry

#: The `meaning_req` axis. Not a label — a request to call 1.
MEANING_REQS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: The `error_spec` axis, ordered by severity. `unreadable` exists to exercise
#: the `correction: null` path: without it the parser's inverted-failure guard
#: never appears in training data.
ERROR_SPECS: tuple[str, ...] = ("none", "one", "few", "many", "unreadable")

#: Specs per batch. Fixed project-wide rather than tunable: a range would leave
#: the Phase 2 parity test measuring an unspecified configuration.
K = 6


@dataclass(frozen=True)
class Sense:
    """One row of `senses_pool.parquet`, as the sampler needs it."""

    sense_uid: str
    target: str
    target_norm: str
    pos: str
    definition: str
    cefr: str | None
    is_multiword: bool
    is_placeholder: bool


@dataclass(frozen=True)
class Batch:
    """One call-1 request: a sense plus the K specs to write for it."""

    batch_uid: str
    sense: Sense
    specs: tuple[DiversifySpec, ...]


@dataclass
class SampleStats:
    """What the sampler actually produced, for the generation report."""

    senses: int = 0
    specs: int = 0
    multiword_senses: int = 0
    meaning_req_counts: dict[int, int] = field(default_factory=dict)
    error_spec_counts: dict[str, int] = field(default_factory=dict)
    profile_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "senses": self.senses,
            "specs": self.specs,
            "multiword_senses": self.multiword_senses,
            "meaning_req_counts": {str(k): v for k, v in sorted(self.meaning_req_counts.items())},
            "error_spec_counts": dict(sorted(self.error_spec_counts.items())),
            "profile_counts": dict(sorted(self.profile_counts.items())),
        }


def spec_uid(sense_uid: str, meaning_req: int, error_spec: str, profile_id: str, seed: int) -> str:
    """Stable id for one (sense, cell, profile) request.

    Content-addressed so the cache and the resume store agree on identity: the
    same request produces the same id on a re-run, which is what makes an
    interrupted generation resumable without a side ledger.
    """
    payload = f"{sense_uid}|{meaning_req}|{error_spec}|{profile_id}|{seed}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def batch_uid(sense_uid: str, spec_ids: Sequence[str]) -> str:
    """Stable id for a batch — the cache key for one call-1 request."""
    payload = "|".join([sense_uid, *spec_ids])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _weighted_cells(
    rng: random.Random,
    count: int,
    meaning_weights: dict[int, float],
    error_weights: dict[str, float],
) -> list[tuple[int, str]]:
    """Draw `count` distinct grid cells, weighted, without replacement.

    Distinct is the point: two identical cells in one batch ask the model for the
    same sentence twice. When `count` exceeds the grid size, cells repeat — which
    only happens if K is raised past 25.
    """
    cells = [(m, e) for m in MEANING_REQS for e in ERROR_SPECS]
    weights = [meaning_weights.get(m, 1.0) * error_weights.get(e, 1.0) for m, e in cells]

    chosen: list[tuple[int, str]] = []
    pool = list(zip(cells, weights))
    while len(chosen) < count:
        if not pool:
            pool = list(zip(cells, weights))
        total = sum(weight for _, weight in pool)
        if total <= 0:
            chosen.extend(cell for cell, _ in pool[: count - len(chosen)])
            break
        draw = rng.random() * total
        cumulative = 0.0
        for index, (cell, weight) in enumerate(pool):
            cumulative += weight
            if draw <= cumulative:
                chosen.append(cell)
                pool.pop(index)
                break
        else:  # pragma: no cover - float drift only
            chosen.append(pool.pop()[0])
    return chosen


def _profile_for(rng: random.Random, registry: ProfileRegistry, meaning_req: int) -> Profile:
    """Pick a profile that can plausibly produce the requested meaning band.

    A near-native profile writes fluent English by construction, so asking it for
    `meaning_req = 0` yields either a clean sentence (and the cell is wasted) or a
    contradiction the model resolves by ignoring one instruction. Restricting the
    pairing keeps the request coherent.
    """
    if meaning_req >= 3:
        candidates = registry.profiles
    else:
        candidates = tuple(p for p in registry.profiles if not p.is_near_native)
        if not candidates:  # pragma: no cover - registry guarantees non-native profiles
            candidates = registry.profiles
    return rng.choice(list(candidates))


def _ranked(senses: Sequence[Sense], seed: int) -> list[Sense]:
    """Every sense in a stable seeded order, independent of how many are wanted.

    A per-sense hash rather than `rng.sample`: `sample(pool, n)` draws a
    *combination*, so its result depends on `n`. Ranking once and slicing means
    the first 84 of a 1000-sense draw are exactly the 84 a smaller draw returned,
    which is what lets a dataset grow without re-paying for what it already has.
    """
    return sorted(
        senses,
        key=lambda sense: hashlib.sha256(f"{seed}|{sense.sense_uid}".encode()).hexdigest(),
    )


def sample_senses(
    senses: Sequence[Sense],
    count: int,
    *,
    seed: int,
    multiword_share: float = 0.15,
) -> list[Sense]:
    """Draw `count` senses, holding multiword targets to a target share.

    Multiword senses are a fixed fraction of the pool, and eval reports them
    separately, so their share is set here rather than left to chance: too few and
    the Phase 8 subgroup has no usable sample size, too many and the dataset stops
    resembling the product's traffic.

    **Nested in `count`.** The result for a smaller `count` is a prefix of the
    result for a larger one, within each of the two strata. This is what makes
    incremental generation affordable: raising `sample.pilot_senses` from 84 to
    1000 re-draws the same first 84 senses, whose `batch_uid`s are unchanged, so
    the response cache serves them and only the new senses cost calls. Drawing a
    fresh combination per size instead would discard almost everything already
    paid for — measured at 14 of 84 surviving a 84 -> 1000 change.

    Ordering depends only on the seed and each sense's own `sense_uid`, so adding
    rows to the pool does not reshuffle the senses already drawn, and Parquet row
    order never enters.
    """
    if count <= 0:
        return []

    ordered = _ranked(senses, seed)
    multiword = [s for s in ordered if s.is_multiword]
    single = [s for s in ordered if not s.is_multiword]

    want_multi = min(len(multiword), round(count * multiword_share))
    want_single = min(len(single), count - want_multi)
    # A pool short on one kind spends the remainder on the other rather than
    # returning fewer senses than asked for.
    want_multi = min(len(multiword), count - want_single)

    # Slicing the two ranked strata keeps the prefix property per stratum. The
    # concatenation order is deterministic, and `build_batches` derives each
    # sense's RNG stream from its own uid, so no downstream result depends on it.
    return multiword[:want_multi] + single[:want_single]


def build_batches(
    senses: Sequence[Sense],
    registry: ProfileRegistry,
    *,
    seed: int,
    k: int = K,
    meaning_weights: dict[int, float] | None = None,
    error_weights: dict[str, float] | None = None,
) -> tuple[list[Batch], SampleStats]:
    """Turn sampled senses into call-1 batches of K specs each.

    Deterministic given `(senses, registry, seed)`: each sense derives its own RNG
    stream from the seed and its `sense_uid`, so adding a sense to the pool does
    not reshuffle the specs of every other sense — which would invalidate a cache
    built by an earlier run.
    """
    mw = meaning_weights or dict.fromkeys(MEANING_REQS, 1.0)
    ew = error_weights or dict.fromkeys(ERROR_SPECS, 1.0)

    batches: list[Batch] = []
    stats = SampleStats()

    for sense in senses:
        # Per-sense stream: `seed` alone would make every sense's cells depend on
        # its position in the list.
        stream = random.Random(f"{seed}|{sense.sense_uid}")
        cells = _weighted_cells(stream, k, mw, ew)

        specs: list[DiversifySpec] = []
        for meaning_req, error_spec in cells:
            profile = _profile_for(stream, registry, meaning_req)
            specs.append(
                DiversifySpec(
                    spec_id=spec_uid(sense.sense_uid, meaning_req, error_spec, profile.id, seed),
                    profile_id=profile.id,
                    meaning_req=meaning_req,
                    error_spec=error_spec,
                    error_bias=profile.error_bias,
                )
            )
            stats.specs += 1
            stats.meaning_req_counts[meaning_req] = stats.meaning_req_counts.get(meaning_req, 0) + 1
            stats.error_spec_counts[error_spec] = stats.error_spec_counts.get(error_spec, 0) + 1
            stats.profile_counts[profile.id] = stats.profile_counts.get(profile.id, 0) + 1

        batches.append(
            Batch(
                batch_uid=batch_uid(sense.sense_uid, [s.spec_id for s in specs]),
                sense=sense,
                specs=tuple(specs),
            )
        )
        stats.senses += 1
        if sense.is_multiword:
            stats.multiword_senses += 1

    return batches, stats


#: `batch_specs.parquet` columns. One row per spec, batch membership by column.
SPEC_COLUMNS: tuple[str, ...] = (
    "spec_id",
    "batch_uid",
    "sense_uid",
    "target",
    "target_norm",
    "pos",
    "definition",
    "cefr",
    "is_multiword",
    "is_placeholder",
    "profile_id",
    "meaning_req",
    "error_spec",
    "seed",
)


def spec_rows(batches: Sequence[Batch], seed: int) -> Iterator[dict[str, Any]]:
    """Flatten batches into `batch_specs.parquet` rows."""
    for batch in batches:
        for spec in batch.specs:
            yield {
                "spec_id": spec.spec_id,
                "batch_uid": batch.batch_uid,
                "sense_uid": batch.sense.sense_uid,
                "target": batch.sense.target,
                "target_norm": batch.sense.target_norm,
                "pos": batch.sense.pos,
                "definition": batch.sense.definition,
                "cefr": batch.sense.cefr,
                "is_multiword": batch.sense.is_multiword,
                "is_placeholder": batch.sense.is_placeholder,
                "profile_id": spec.profile_id,
                "meaning_req": spec.meaning_req,
                "error_spec": spec.error_spec,
                "seed": seed,
            }


def read_pool(path: str | Path) -> list[Sense]:
    """Load `senses_pool.parquet` into `Sense` records."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=[
            "sense_uid",
            "target",
            "target_norm",
            "pos",
            "definition",
            "cefr",
            "is_multiword",
            "is_placeholder",
        ],
    )
    return [Sense(**row) for row in table.to_pylist()]


def write_specs(batches: Sequence[Batch], seed: int, path: str | Path) -> int:
    """Write `batch_specs.parquet` with a pinned schema. Returns the row count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("spec_id", pa.string()),
            pa.field("batch_uid", pa.string()),
            pa.field("sense_uid", pa.string()),
            pa.field("target", pa.string()),
            pa.field("target_norm", pa.string()),
            pa.field("pos", pa.string()),
            pa.field("definition", pa.string()),
            pa.field("cefr", pa.string()),
            pa.field("is_multiword", pa.bool_()),
            pa.field("is_placeholder", pa.bool_()),
            pa.field("profile_id", pa.string()),
            pa.field("meaning_req", pa.int32()),
            pa.field("error_spec", pa.string()),
            pa.field("seed", pa.int64()),
        ]
    )
    rows = list(spec_rows(batches, seed))
    table = pa.Table.from_pydict(
        {name: [row[name] for row in rows] for name in SPEC_COLUMNS},
        schema=schema,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy", row_group_size=50_000)
    return len(rows)


def load_weights(params: dict[str, Any]) -> tuple[dict[int, float], dict[str, float]]:
    """Read the two weight tables out of a parsed `params.yaml` `sample:` block.

    Keys arrive as strings from YAML/JSON; `meaning_req` is an int everywhere
    else, so they are coerced here rather than at every lookup.
    """
    meaning = {int(k): float(v) for k, v in (params.get("meaning_weights") or {}).items()}
    error = {str(k): float(v) for k, v in (params.get("error_spec_weights") or {}).items()}
    return (meaning or dict.fromkeys(MEANING_REQS, 1.0), error or dict.fromkeys(ERROR_SPECS, 1.0))


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a report deterministically: sorted keys, trailing newline, no clock."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ERROR_SPECS",
    "K",
    "MEANING_REQS",
    "SPEC_COLUMNS",
    "Batch",
    "SampleStats",
    "Sense",
    "batch_uid",
    "build_batches",
    "load_weights",
    "read_pool",
    "sample_senses",
    "spec_rows",
    "spec_uid",
    "write_report",
    "write_specs",
]
