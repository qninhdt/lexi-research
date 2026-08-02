"""The data stages, as functions a DVC stage and a human type the same way.

Each returns a report dict which is written next to its output and logged to the
run. Reports carry counts and rates only — never dictionary text — so they stay
in Git as the provenance record while the parquet stays private.

Generation and labelling reach a teacher endpoint; the rest are pure. That split
is why the first two are declared in `dvc.yaml` but never run in CI.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lexi_research.cli.config import Config
from lexi_research.format import BandConfig, default_config_path
from lexi_research.format.bands import MAX_BAND, MIN_BAND, count_words, penalty
from lexi_research.format.parser import ParseOk, parse_correction
from lexi_research.format.tags import TagGroup
from lexi_research.teacher.schemas import DiversifySpec

from .generate import generate_batches, text_rows, write_texts
from .jsonl_store import JsonlStore
from .label import label_rows, label_texts, read_texts_for_labelling, write_labels
from .process import process_parquet
from .profiles import load_profiles
from .sample_batches import (
    Batch,
    Sense,
    build_batches,
    load_weights,
    read_pool,
    sample_senses,
    write_report,
    write_specs,
)


class StageError(RuntimeError):
    """A stage could not run, or produced something its gate rejects."""


def _teacher_client(config: Config, cache_dir: Path) -> Any:
    from lexi_research.teacher import ResponseCache, TeacherClient, TeacherConfig, prompt_hash

    try:
        teacher = TeacherConfig.from_env(dict(os.environ))
    except ValueError as exc:
        raise StageError(str(exc)) from exc
    return TeacherClient(
        teacher,
        cache=ResponseCache(cache_dir),
        prompt_hash=prompt_hash(),
    )


def run_sample(config: Config, *, pool: str | Path, out: str | Path, full: bool) -> dict[str, Any]:
    """Draw senses and build call-1 batches. Deterministic given the seed."""
    section = dict(config.section("sample"))
    senses = read_pool(pool)
    count = int(section["full_senses" if full else "pilot_senses"])
    picked = sample_senses(
        senses,
        count,
        seed=int(section["seed"]),
        multiword_share=float(section["multiword_share"]),
    )
    if not picked:
        raise StageError(f"sampled no senses from a pool of {len(senses)}")

    meaning_weights, error_weights = load_weights(section)
    batches, stats = build_batches(
        picked,
        load_profiles(),
        seed=int(section["seed"]),
        k=int(section["k"]),
        meaning_weights=meaning_weights,
        error_weights=error_weights,
    )
    out_dir = Path(out)
    rows = write_specs(batches, int(section["seed"]), out_dir / "batch_specs.parquet")
    report = {
        "pool_senses": len(senses),
        "batches": len(batches),
        "specs": rows,
        **stats.as_dict(),
    }
    write_report(out_dir / "sample-report.json", report)
    return report


def run_generate(
    config: Config, *, specs: str | Path, out: str | Path, cache: str | Path
) -> dict[str, Any]:
    """Call 1: turn specs into learner sentences. Resumable, cache-first."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = batches_from_specs(specs)
    client = _teacher_client(config, Path(cache))
    store = JsonlStore(out_dir / "generate.jsonl")

    stats = asyncio.run(
        generate_batches(
            batches,
            client,
            store,
            load_profiles().traits_map(),
            min_distinct2=config.get_float("generate.min_distinct2"),
        )
    )
    written = write_texts(text_rows(store), out_dir / "raw_texts.parquet")
    report = {**stats.as_dict(), "texts_written": written, "cost": round(client.stats.cost, 4)}
    write_report(out_dir / "generate-report.json", report)
    return report


def batches_from_specs(path: str | Path) -> list[Batch]:
    """Rebuild sampler batches from the parquet the sample stage wrote.

    The specs file is the boundary between sampling and generation, so generation
    resumes from it rather than from a re-run of the sampler — which would have to
    reproduce the same RNG stream to agree.
    """
    import pyarrow.parquet as pq

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pq.read_table(path).to_pylist():
        grouped.setdefault(str(row["batch_uid"]), []).append(row)

    batches: list[Batch] = []
    for batch_uid, rows in sorted(grouped.items()):
        head = rows[0]
        sense = Sense(
            sense_uid=head["sense_uid"],
            target=head["target"],
            target_norm=head["target_norm"],
            pos=head["pos"],
            definition=head["definition"],
            cefr=head["cefr"],
            is_multiword=head["is_multiword"],
            is_placeholder=head["is_placeholder"],
        )
        specs = tuple(
            DiversifySpec(
                spec_id=row["spec_id"],
                profile_id=row["profile_id"],
                meaning_req=int(row["meaning_req"]),
                error_spec=row["error_spec"],
            )
            for row in sorted(rows, key=lambda item: str(item["spec_id"]))
        )
        batches.append(Batch(batch_uid=batch_uid, sense=sense, specs=specs))
    return batches


def run_label(
    config: Config, *, texts: str | Path, out: str | Path, cache: str | Path
) -> dict[str, Any]:
    """Call 2: grade every sentence with the prompt the student will be served."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_texts_for_labelling(texts)
    client = _teacher_client(config, Path(cache))
    store = JsonlStore(out_dir / "label.jsonl")

    stats = asyncio.run(label_texts(rows, client, store))
    written = write_labels(label_rows(store), out_dir / "raw_labels.parquet")
    report = {
        "texts": len(rows),
        "requests": stats.requests,
        "cached": stats.cached,
        "failed": stats.failed,
        "labelled": stats.labelled,
        "rejected": stats.rejected,
        "reject_reasons": dict(sorted(stats.reject_reasons.items())),
        "validity_rate": round(stats.validity_rate, 4),
        "null_corrections": stats.null_corrections,
        "labels_written": written,
        "cost": round(client.stats.cost, 4),
    }
    write_report(out_dir / "label-report.json", report)
    return report


def run_gec_import(
    config: Config, *, corpus: str | Path, out: str | Path
) -> dict[str, Any]:
    """Stage A: convert a human-annotated learner corpus into the edit format.

    Pure and free. Unlike `generate` and `label` it reaches no endpoint, so it can
    run in CI and its output does not depend on a provider being up.

    The rows it writes carry `correction` but neither `meaning` nor `feedback`,
    which is why they go to their own directory and their own parquet rather than
    joining `data/clean/`: a row missing two of the three answer fields would fail
    `validate_output` and has no business in the artifact stage B trains on.
    """
    from .gec_import import import_corpus

    section = dict(config.section("gec"))
    rows, report = import_corpus(
        corpus,
        seed=int(section["seed"]),
        max_stratum_share=float(section["max_stratum_share"]),
        val_share=float(section["val_share"]),
        min_words=int(section["min_words"]),
        min_conversion_rate=float(section["min_conversion_rate"]),
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            raise StageError(f"the {split} split is empty; check gec.val_share")
        pq.write_table(pa.Table.from_pylist(subset), out_dir / f"{split}.parquet")

    report["corpus_sha256"] = _corpus_digest(corpus)
    report["corrector_prompt_hash"] = _corrector_prompt_hash()
    write_report(out_dir / "gec-import-report.json", report)
    return report


def _corpus_digest(corpus: str | Path) -> str:
    """SHA-256 over the M2 files, so a corpus swap invalidates the artifact.

    Hashes the files that were read rather than the directory: the tarball also
    ships JSON and licence text, and a change to those does not change the rows.
    """
    from .gec_import import TRAIN_FILES

    digest = hashlib.sha256()
    for name, _ in TRAIN_FILES:
        path = Path(corpus) / "m2" / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _corrector_prompt_hash() -> str:
    """Recorded in the report so a prompt edit is visible in the artifact's lineage."""
    from lexi_research.train.corrector_prompt import corrector_prompt_hash

    return corrector_prompt_hash()


def run_process(
    config: Config,
    *,
    texts: str | Path,
    labels: str | Path,
    out: str | Path,
    band_config: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, balance and split in one pass.

    One stage rather than three: `process_rows` is a single transformation whose
    intermediate results nothing else consumes, and its report already breaks out
    reject reasons, the distribution before and after balancing, and the
    contamination check. Three stages would buy three parquet round-trips.
    """
    config_path = band_config or default_config_path()
    report = process_parquet(
        texts,
        labels,
        out,
        BandConfig.from_json(config_path),
        seed=config.get_int("split.seed"),
        version=config.get_str("split.version"),
        max_stratum_share=config.get_float("balance.max_stratum_share"),
    )
    rejected = int(report["rejected_rows"])
    total = int(report["input_rows"]) or 1
    ceiling = config.get_float("validate.max_reject_rate")
    if rejected / total > ceiling:
        raise StageError(
            f"{rejected} of {total} rows ({rejected / total:.1%}) failed validation, "
            f"over the {ceiling:.1%} ceiling. At this rate the prompt or the schema "
            "is at fault, not the data."
        )
    crossing = int(report["contamination"].get("cross_split_text_hashes", 0))
    if crossing:
        raise StageError(
            f"{crossing} distinct texts appear in more than one split; the grouped "
            "split by target word did not hold, so test scores would be inflated"
        )
    return report


def _group_penalties(
    rows: Sequence[Mapping[str, Any]], config: BandConfig
) -> dict[TagGroup, list[float]]:
    """Per-row penalty in each group, over rows whose correction parses."""
    found: dict[TagGroup, list[float]] = {group: [] for group in TagGroup}
    for row in rows:
        correction = row.get("correction")
        if correction is None:
            continue
        parsed = parse_correction(str(correction))
        if not isinstance(parsed, ParseOk):
            continue
        words = count_words(str(row["text"]))
        for group in TagGroup:
            found[group].append(penalty(parsed.edits, group, words, config))
    return found


def _quantiles(values: Sequence[float], fractions: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise StageError("no parseable corrections to calibrate against")
    picked = []
    for fraction in fractions:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        picked.append(ordered[index])
    return picked


def run_calibrate(
    config: Config,
    *,
    rows_path: str | Path,
    out: str | Path,
    band_config: str | Path | None = None,
) -> dict[str, Any]:
    """Place the band cut points on the real penalty distribution.

    The shipped thresholds are design guesses and `calibrated: false` says so;
    the eval harness refuses to report band metrics until this stage flips it.
    Cut points go at the quantiles that give `calibrate.target: uniform` an even
    share per band — a band so rare the student never sees it is not a band.

    Both derived bands share one threshold vector, so the quantiles are taken
    over the pooled penalties rather than fitted per group.
    """
    import pyarrow.parquet as pq

    source = Path(band_config or default_config_path())
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    current = BandConfig.from_dict(payload)

    target = config.get_str("calibrate.target")
    if target != "uniform":
        raise StageError(f"calibrate.target={target!r} is not implemented; expected 'uniform'")

    rows: list[dict[str, Any]] = pq.read_table(rows_path).to_pylist()
    by_group = _group_penalties(rows, current)
    pooled = [value for values in by_group.values() for value in values]
    bands = MAX_BAND - MIN_BAND
    # Thresholds ascend and the band falls as the penalty rises, so the cut for
    # the top band is the lowest quantile: band 4 is the cleanest fifth.
    fractions = [index / (bands + 1) for index in range(1, bands + 1)]
    thresholds = _quantiles(pooled, fractions)

    payload["thresholds"] = thresholds
    payload["calibrated"] = True
    payload["version"] = int(payload["version"]) + 1
    calibrated = BandConfig.from_dict(payload)

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "version": payload["version"],
        "rows": len(rows),
        "parseable_corrections": len(by_group[TagGroup.CORRECTNESS]),
        "thresholds": thresholds,
        "band_distribution": {
            group.value: dict(
                sorted(Counter(calibrated.band_of(value) for value in by_group[group]).items())
            )
            for group in TagGroup
        },
    }
    write_report(destination.parent / "calibrate-report.json", report)
    return report


__all__ = [
    "StageError",
    "run_calibrate",
    "run_gec_import",
    "run_generate",
    "run_label",
    "run_process",
    "run_sample",
]
