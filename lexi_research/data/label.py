"""Call 2 — the teacher. Grades generated sentences with the inference prompt.

This is the stage that makes the project distillation rather than training on a
transcript. It sees only `{target, sense, text}`: never the profile, never the
requested meaning band, never the error recipe. Its output is therefore a
function of the sentence, which is exactly the function the student is asked to
learn.

The alternative — asking one call to both write and label — produces labels that
describe the *instruction* instead of the text, and the defect is invisible
afterwards because a correct row and a wrong one look identical.

Three properties do the real work:

**Prompt parity.** Every grading goes through `render_grader_prompt`, one text at
a time. That is byte-for-byte the prompt the trainer and the server use. Batching
call 2 would save money and cost the thing the phase exists to establish, so the
saving is declined here and taken on call 1 instead, where the prompt is not the
contract.

**Bands are derived.** `grammar` and `naturalness` never come from the model.
`validate_output` computes them from the correction's tags, so an identical error
set always scores identically and the thresholds stay retunable without
retraining.

**Resume.** Each grading is appended to a JSONL log keyed by `req_uid` as it
lands, so an interrupted run restarts having kept everything it paid for.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexi_research.format import BandConfig, ValidationError, default_config_path, validate_output
from lexi_research.teacher import (
    GraderOutput,
    RetryExhausted,
    TeacherClient,
    render_grader_prompt,
)
from lexi_research.teacher.schemas import SenseRef

from .jsonl_store import JsonlStore

#: Columns of `raw_labels.parquet`. Joined to `raw_texts.parquet` on `req_uid`.
#:
#: The spec columns (`meaning_req`, `error_spec`, `profile_id`) are deliberately
#: absent: they live on the text side, and keeping them out of the label artifact
#: is what makes "the spec is not a label" checkable rather than aspirational.
#:
#: `prompt_hash` is present because a label is only meaningful against the rubric
#: that produced it. Without it, a run that resumed across a prompt edit yields a
#: dataset holding two rubrics with no way to tell the rows apart — and every
#: aggregate over it silently averages the two.
LABEL_COLUMNS: tuple[str, ...] = (
    "req_uid",
    "sense_uid",
    "text",
    "correction",
    "meaning",
    "feedback",
    "grammar",
    "naturalness",
    "tags",
    "n_edits",
    "n_words",
    "pass_index",
    "prompt_hash",
)

#: The fields call 2 is allowed to read off a text row. Anything else on the row
#: is spec metadata, and reading it here would be the leak this phase guards
#: against — so the allowlist is enforced rather than documented.
_GRADED_FIELDS: frozenset[str] = frozenset(
    {"req_uid", "sense_uid", "target", "pos", "definition", "text"}
)


@dataclass
class LabelStats:
    """What a call-2 run produced, for `generation-report.json`."""

    requests: int = 0
    cached: int = 0
    failed: int = 0
    labelled: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    meaning_counts: dict[int, int] = field(default_factory=dict)
    tag_counts: dict[str, int] = field(default_factory=dict)
    null_corrections: int = 0

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def accept(self, meaning: int, tags: Sequence[str], *, null_correction: bool) -> None:
        self.labelled += 1
        self.meaning_counts[meaning] = self.meaning_counts.get(meaning, 0) + 1
        for tag in tags:
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1
        if null_correction:
            self.null_corrections += 1

    @property
    def validity_rate(self) -> float:
        """Share of gradings that passed the six checks — pilot gate G3."""
        total = self.labelled + self.rejected
        return self.labelled / total if total else 0.0

    @property
    def other_tag_share(self) -> float:
        """Share of `other` among all tags — pilot gate G5.

        A high value means the taxonomy is missing a category the teacher keeps
        reaching for, which is a Phase 1 problem rather than a prompt problem.
        """
        total = sum(self.tag_counts.values())
        return self.tag_counts.get("other", 0) / total if total else 0.0

    @property
    def middle_band_share(self) -> float:
        """Share of `meaning` in {1,2,3} — pilot gate G2.

        The bimodal failure this catches cannot be repaired downstream: a dataset
        holding only bands 0 and 4 teaches a model that cannot grade the middle,
        which is where real learners land.
        """
        total = sum(self.meaning_counts.values())
        if not total:
            return 0.0
        return sum(self.meaning_counts.get(band, 0) for band in (1, 2, 3)) / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cached": self.cached,
            "failed": self.failed,
            "labelled": self.labelled,
            "rejected": self.rejected,
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "meaning_counts": {str(k): v for k, v in sorted(self.meaning_counts.items())},
            "tag_counts": dict(sorted(self.tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "null_corrections": self.null_corrections,
            "validity_rate": round(self.validity_rate, 4),
            "other_tag_share": round(self.other_tag_share, 4),
            "middle_band_share": round(self.middle_band_share, 4),
        }


def graded_view(row: dict[str, Any]) -> dict[str, Any]:
    """The subset of a text row call 2 is allowed to see.

    Narrowing here rather than trusting the caller is what makes spec isolation a
    property of the code: this stage cannot leak `meaning_req` into a label
    because it never holds it.
    """
    return {name: row[name] for name in _GRADED_FIELDS if name in row}


def _cache_identity(row: dict[str, Any], pass_index: int) -> dict[str, Any]:
    """Stable identity for one grading request.

    The rendered prompt carries a fresh nonce per call, so it cannot be the key.
    `pass_index` participates because the self-consistency re-grade must genuinely
    re-ask the teacher rather than read its own first answer back.
    """
    return {
        "stage": "grade",
        "pass": pass_index,
        "req_uid": row["req_uid"],
        "target": row["target"],
        "text": row["text"],
    }


def _log_id(req_uid: str, pass_index: int) -> str:
    """Resume key for a grading. Pass 1 must not read pass 0's row as done."""
    return req_uid if pass_index == 0 else f"{req_uid}#p{pass_index}"


async def grade_one(
    row: dict[str, Any],
    client: TeacherClient,
    *,
    pass_index: int = 0,
) -> GraderOutput | None:
    """Grade one sentence with the frozen inference prompt.

    Returns `None` when every retry failed: one unlucky row must not abort a run
    that has already paid for thousands of others.

    The prompt is passed as a renderer rather than a rendered list so each attempt
    carries a fresh nonce. Measured on this proxy, about 7% of gradings return
    empty tool arguments, and re-asking with identical bytes at temperature 0
    leaves a residue of rows that fail every attempt; re-rendering recovered all
    of them.
    """
    view = graded_view(row)
    sense = SenseRef(definition=view["definition"], pos=view["pos"])

    def messages() -> list[Any]:
        return render_grader_prompt(view["target"], sense, view["text"])

    try:
        return await client.call(
            messages, GraderOutput, cache_extra=_cache_identity(view, pass_index)
        )
    except RetryExhausted:
        return None


def label_row(
    row: dict[str, Any],
    output: GraderOutput,
    config: BandConfig,
    stats: LabelStats,
    *,
    pass_index: int = 0,
    prompt_hash: str = "",
) -> dict[str, Any] | None:
    """Validate a grading and build its label row, or count why it was rejected."""
    payload = {
        "correction": output.correction,
        "meaning": output.meaning,
        "feedback": output.feedback,
    }
    result = validate_output(payload, row["text"], config)
    if isinstance(result, ValidationError):
        stats.reject(result.code)
        return None

    edits = result.edits
    tags = sorted({edit.tag for edit in edits}) if edits else []
    stats.accept(result.meaning, tags, null_correction=edits is None)

    return {
        "req_uid": _log_id(str(row["req_uid"]), pass_index),
        "text_uid": row["req_uid"],
        "sense_uid": row["sense_uid"],
        "text": row["text"],
        "correction": output.correction,
        "meaning": result.meaning,
        "feedback": result.feedback,
        "grammar": result.bands.grammar,
        "naturalness": result.bands.naturalness,
        "tags": tags,
        "n_edits": len(edits) if edits is not None else 0,
        "n_words": len(str(row["text"]).split()),
        "pass_index": pass_index,
        "prompt_hash": prompt_hash,
    }


class PromptMismatch(RuntimeError):
    """The resume log holds labels written under a different grading rubric.

    Raised rather than silently continuing because `label_texts` skips by log id
    *before* the response cache is consulted. A prompt edit therefore does not
    invalidate work the log already covers: the run would grade only the rows that
    happen to be missing, and the artifact would hold two rubrics with no marker
    distinguishing them. Every aggregate over such a file — the band distribution,
    the gate, the ceiling — silently averages across both.
    """

    def __init__(self, found: set[str], current: str) -> None:
        known = ", ".join(sorted(value[:12] or "<unrecorded>" for value in found))
        super().__init__(
            f"data/raw/label.jsonl holds labels graded under prompt(s) [{known}] but the "
            f"current prompt is {current[:12]}. Delete the log to re-grade every row "
            f"under one rubric, or restore the previous prompt. Resuming would mix "
            f"two rubrics in one dataset with no way to tell the rows apart."
        )
        self.found = found
        self.current = current


def _assert_one_prompt(store: JsonlStore, prompt_hash: str) -> None:
    """Refuse to extend a log written under a different rubric."""
    found = {
        str(record.get("prompt_hash", ""))
        for record in store.read()
        if "meaning" in record
    }
    if found and found != {prompt_hash}:
        raise PromptMismatch(found, prompt_hash)


async def label_texts(
    rows: Sequence[dict[str, Any]],
    client: TeacherClient,
    store: JsonlStore,
    *,
    config: BandConfig | None = None,
    pass_index: int = 0,
    concurrency: int | None = None,
) -> LabelStats:
    """Grade every text, appending each label as it lands.

    One call per sentence, deliberately: this prompt is the inference contract,
    and a batched variant of it would not be the prompt the student runs.

    Refuses to extend a log whose labels were graded under a different prompt.
    Resume skips by log id before the cache is reached, so a prompt edit would
    otherwise leave most rows untouched and produce a two-rubric dataset.
    """
    band_config = config if config is not None else BandConfig.from_json(default_config_path())
    stats = LabelStats()
    prompt_hash = client.prompt_hash
    _assert_one_prompt(store, prompt_hash)

    done = store.completed_ids()
    pending = [row for row in rows if _log_id(str(row["req_uid"]), pass_index) not in done]
    stats.cached = len(rows) - len(pending)

    limit = asyncio.Semaphore(concurrency or client.config.concurrency)
    write_lock = asyncio.Lock()

    async def run(row: dict[str, Any]) -> None:
        async with limit:
            output = await grade_one(row, client, pass_index=pass_index)
        async with write_lock:
            stats.requests += 1
            if output is None:
                stats.failed += 1
                return
            label = label_row(
                row,
                output,
                band_config,
                stats,
                pass_index=pass_index,
                prompt_hash=prompt_hash,
            )
            if label is not None:
                store.append(label)

    await asyncio.gather(*(run(row) for row in pending))
    return stats


def label_rows(store: JsonlStore) -> list[dict[str, Any]]:
    """Label rows from the log, keyed back to the text they graded."""
    out: list[dict[str, Any]] = []
    for record in store.latest_by_id().values():
        if "meaning" not in record:
            continue
        row = dict(record)
        row["req_uid"] = row.pop("text_uid", row["req_uid"])
        out.append(row)
    return out


async def self_consistency_pairs(
    rows: Sequence[dict[str, Any]],
    client: TeacherClient,
    *,
    sample: int = 200,
    seed: int = 0,
    concurrency: int | None = None,
    config: BandConfig | None = None,
) -> list[dict[str, Any]]:
    """Re-grade a shuffled sample for pilot gate G1.

    Two details are load-bearing. The order is shuffled, so a teacher that is
    merely consistent *within a sequence* — anchoring on what it just said —
    does not pass by accident. And `pass_index=1` keeps the request out of pass
    0's cache entry, so the teacher actually answers again; pass the client a
    `NullCache` as well when the run's own cache is on disk.

    Returns one row per sampled text carrying both readings, which is what the
    QWK and edit-F1 computation in the gate report consumes.
    """
    if sample <= 0 or not rows:
        return []

    rng = random.Random(seed)
    # This gate compares against *accepted* first-pass labels.  Calling it with
    # raw text rows would silently make the measurement undefined.
    eligible = [
        row
        for row in rows
        if isinstance(row.get("meaning"), int)
        and "correction" in row
        and isinstance(row.get("text"), str)
    ]
    if not eligible:
        return []
    ordered = sorted(eligible, key=lambda row: str(row["req_uid"]))
    picked = rng.sample(ordered, min(sample, len(ordered)))
    rng.shuffle(picked)

    limit = asyncio.Semaphore(concurrency or client.config.concurrency)
    band_config = config if config is not None else BandConfig.from_json(default_config_path())

    async def regrade(row: dict[str, Any]) -> dict[str, Any] | None:
        async with limit:
            output = await grade_one(row, client, pass_index=1)
        if output is None:
            return None
        checked = validate_output(output.model_dump(mode="json"), row["text"], band_config)
        if isinstance(checked, ValidationError):
            return None
        return {
            "req_uid": row["req_uid"],
            "text": row["text"],
            "meaning": row["meaning"],
            "correction": row["correction"],
            "meaning_pass2": output.meaning,
            "correction_pass2": output.correction,
        }

    results = await asyncio.gather(*(regrade(row) for row in picked))
    return [row for row in results if row is not None]


def read_texts_for_labelling(path: str | Path) -> list[dict[str, Any]]:
    """Read `raw_texts.parquet`, keeping only the columns call 2 may see.

    Projecting at the read is the strongest form of the isolation rule: the spec
    columns are never loaded, so no later code path can accidentally pass one to
    the grader.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=sorted(_GRADED_FIELDS))
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def write_labels(rows: Sequence[dict[str, Any]], path: str | Path) -> int:
    """Write `raw_labels.parquet` with a pinned schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    types: dict[str, pa.DataType] = {
        "req_uid": pa.string(),
        "sense_uid": pa.string(),
        "text": pa.string(),
        "correction": pa.string(),
        "meaning": pa.int32(),
        "feedback": pa.string(),
        "grammar": pa.int32(),
        "naturalness": pa.int32(),
        "tags": pa.list_(pa.string()),
        "n_edits": pa.int32(),
        "n_words": pa.int32(),
        "pass_index": pa.int32(),
        "prompt_hash": pa.string(),
    }
    schema = pa.schema([pa.field(name, types[name]) for name in LABEL_COLUMNS])
    ordered = sorted(rows, key=lambda row: (str(row["req_uid"]), int(row.get("pass_index", 0))))
    table = pa.Table.from_pydict(
        {name: [row.get(name) for row in ordered] for name in LABEL_COLUMNS},
        schema=schema,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="snappy", row_group_size=50_000)
    return len(ordered)


__all__ = [
    "LABEL_COLUMNS",
    "LabelStats",
    "PromptMismatch",
    "grade_one",
    "graded_view",
    "label_row",
    "label_rows",
    "label_texts",
    "read_texts_for_labelling",
    "self_consistency_pairs",
    "write_labels",
]
