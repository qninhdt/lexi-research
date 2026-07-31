"""Call 1 — the diversifier. Writes learner-like sentences from specs.

This stage knows the spec: which learner, which meaning band to aim for, how much
error to include. That knowledge exists purely to widen the text distribution.
Nothing it produces is a label, and `label.py` never sees the spec.

Resumability is the operational property that matters. Every batch is keyed by a
content-derived `batch_uid` and appended to a JSONL log as soon as it returns, so
a run killed at 80% restarts having paid for 80%.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexi_research.teacher import (
    DiversifyBatch,
    RetryExhausted,
    TeacherClient,
    render_diversify_prompt,
)
from lexi_research.teacher.schemas import SenseRef

from .diversity import BatchDiversity, batch_diversity
from .jsonl_store import JsonlStore
from .sample_batches import Batch

#: Stand-in when a spec names a profile the registry does not hold. Should never
#: happen — the sampler draws profiles from the registry — but rendering with
#: `StrictUndefined` would abort a whole batch over one bad id.
_UNKNOWN_PROFILE = "a language learner"

#: Columns of `raw_texts.parquet`. One row per generated sentence.
TEXT_COLUMNS: tuple[str, ...] = (
    "req_uid",
    "batch_uid",
    "spec_id",
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
    "text",
)


@dataclass
class GenerateStats:
    """What a call-1 run produced, for `generation-report.json`."""

    batches: int = 0
    batches_cached: int = 0
    batches_failed: int = 0
    texts: int = 0
    texts_rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    low_diversity_batches: int = 0
    diversity: list[BatchDiversity] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.texts_rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        distinct2 = [d.distinct2 for d in self.diversity]
        distinct3 = [d.distinct3 for d in self.diversity]
        return {
            "batches": self.batches,
            "batches_cached": self.batches_cached,
            "batches_failed": self.batches_failed,
            "texts": self.texts,
            "texts_rejected": self.texts_rejected,
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "low_diversity_batches": self.low_diversity_batches,
            "mean_distinct2": round(sum(distinct2) / len(distinct2), 4) if distinct2 else 0.0,
            "mean_distinct3": round(sum(distinct3) / len(distinct3), 4) if distinct3 else 0.0,
        }


def _sense_ref(batch: Batch) -> SenseRef:
    return SenseRef(definition=batch.sense.definition, pos=batch.sense.pos)


def _cache_identity(batch: Batch) -> dict[str, Any]:
    """Stable identity for a call-1 request.

    The rendered prompt cannot be the cache key: `render_diversify_prompt` is
    deterministic today, but keying on the request's meaning rather than on its
    wording means a cosmetic template edit does not silently reuse text written
    under different instructions — the prompt hash, which is part of the key,
    handles that.
    """
    return {
        "stage": "diversify",
        "sense_uid": batch.sense.sense_uid,
        "target": batch.sense.target,
        "definition": batch.sense.definition,
        "specs": [
            {
                "spec_id": spec.spec_id,
                "profile_id": spec.profile_id,
                "meaning_req": spec.meaning_req,
                "error_spec": spec.error_spec,
            }
            for spec in batch.specs
        ],
    }


def validate_text(text: str) -> str | None:
    """Reason to reject one generated sentence on shape alone, or `None` if usable.

    Per-element rather than per-batch: five good sentences must not be discarded
    because the sixth came back empty. Target presence is a separate check
    (`target_is_present`) because it does not apply to every cell.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) < 3:
        return "too_short"
    if len(stripped) > 600:
        return "too_long"
    if "\n" in stripped:
        return "multiline"
    return None


def target_is_present(text: str, target: str) -> bool:
    """True when any word of the target appears in the text, prefix-matched.

    Deliberately loose: learners inflect (`brightly`, `ran`) and multiword targets
    get split up. A strict match would reject exactly the realistic attempts this
    dataset needs, so the check only catches a sentence that ignored the target
    entirely.
    """
    lowered = text.lower()
    words = [word for word in target.lower().split() if word not in {"sb", "sth"}]
    if not words:
        return True
    stems = [word[:4] if len(word) > 5 else word for word in words]
    return any(stem in lowered for stem in stems)


async def generate_batches(
    batches: Sequence[Batch],
    client: TeacherClient,
    store: JsonlStore,
    traits: dict[str, str],
    *,
    min_distinct2: float = 0.7,
    concurrency: int | None = None,
) -> GenerateStats:
    """Run call 1 over `batches`, appending each result as it lands.

    `traits` maps `profile_id -> trait text`, from the profile registry. Passed in
    rather than read from the sampler's `Batch`, which would mean carrying prose
    through Parquet for no reason.

    Already-recorded batches are skipped before any request is built, so a resumed
    run does not even render prompts for work it has done.
    """
    stats = GenerateStats()
    done = store.completed_ids()
    pending = [batch for batch in batches if batch.batch_uid not in done]
    stats.batches_cached = len(batches) - len(pending)

    limit = asyncio.Semaphore(concurrency or client.config.concurrency)
    write_lock = asyncio.Lock()

    async def run(batch: Batch) -> None:
        messages = render_diversify_prompt(
            batch.sense.target,
            _sense_ref(batch),
            batch.specs,
            {
                spec.profile_id: traits.get(spec.profile_id, _UNKNOWN_PROFILE)
                for spec in batch.specs
            },
        )
        async with limit:
            try:
                result = await client.call(
                    messages, DiversifyBatch, cache_extra=_cache_identity(batch)
                )
            except RetryExhausted:
                stats.batches_failed += 1
                return

        returned_ids = [sentence.spec_id for sentence in result.sentences]
        expected_ids = {spec.spec_id for spec in batch.specs}
        if len(returned_ids) != len(set(returned_ids)):
            stats.batches_failed += 1
            return
        if set(returned_ids) - expected_ids:
            stats.batches_failed += 1
            return
        by_spec = {sentence.spec_id: sentence.text for sentence in result.sentences}
        accepted: list[dict[str, Any]] = []

        for spec in batch.specs:
            text = by_spec.get(spec.spec_id)
            if text is None:
                stats.reject("missing_spec_id")
                continue
            reason = validate_text(text)
            if reason is not None:
                stats.reject(reason)
                continue
            expects_target = spec.error_spec != "unreadable" and spec.meaning_req > 0
            if expects_target and not target_is_present(text, batch.sense.target):
                stats.reject("target_absent")
                continue
            accepted.append(
                {
                    "req_uid": spec.spec_id,
                    "batch_uid": batch.batch_uid,
                    "spec_id": spec.spec_id,
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
                    "text": text.strip(),
                }
            )

        measure = batch_diversity(
            batch.batch_uid, [row["text"] for row in accepted], threshold=min_distinct2
        )

        async with write_lock:
            for row in accepted:
                store.append(row)
            # The batch marker is written last: its presence means every sentence
            # of the batch is already durable, so a crash between the two leaves a
            # batch that is re-run rather than one silently recorded as complete.
            store.append({"req_uid": batch.batch_uid, "kind": "batch_done"})
            stats.batches += 1
            stats.texts += len(accepted)
            stats.diversity.append(measure)
            if measure.is_collapsed:
                stats.low_diversity_batches += 1

    await asyncio.gather(*(run(batch) for batch in pending))
    return stats


def text_rows(store: JsonlStore) -> list[dict[str, Any]]:
    """Sentence rows from the log, batch markers dropped, last write winning."""
    return [
        record
        for record in store.latest_by_id().values()
        if record.get("kind") != "batch_done" and "text" in record
    ]


def write_texts(rows: Sequence[dict[str, Any]], path: str | Path) -> int:
    """Write `raw_texts.parquet` with a pinned schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    types: dict[str, pa.DataType] = {
        "req_uid": pa.string(),
        "batch_uid": pa.string(),
        "spec_id": pa.string(),
        "sense_uid": pa.string(),
        "target": pa.string(),
        "target_norm": pa.string(),
        "pos": pa.string(),
        "definition": pa.string(),
        "cefr": pa.string(),
        "is_multiword": pa.bool_(),
        "is_placeholder": pa.bool_(),
        "profile_id": pa.string(),
        "meaning_req": pa.int32(),
        "error_spec": pa.string(),
        "text": pa.string(),
    }
    schema = pa.schema([pa.field(name, types[name]) for name in TEXT_COLUMNS])
    ordered = sorted(rows, key=lambda row: str(row["req_uid"]))
    table = pa.Table.from_pydict(
        {name: [row.get(name) for row in ordered] for name in TEXT_COLUMNS},
        schema=schema,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy", row_group_size=50_000)
    return len(ordered)


__all__ = [
    "TEXT_COLUMNS",
    "GenerateStats",
    "generate_batches",
    "target_is_present",
    "text_rows",
    "validate_text",
    "write_texts",
]
