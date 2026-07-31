"""Score a predictions file into a report. CPU only, model never loaded.

Generation and scoring are separate on purpose: a metric fix must not cost a
re-run of the model. `lexi eval predict` writes predictions, this reads them, and
the two meet at a JSONL schema rather than at a shared process.

Two things this refuses to do. It will not report band metrics from an
uncalibrated `band_config.json` — the thresholds shipped with the repo are design
guesses, and a grade derived from a guess is a guess. And it will not report
without a ceiling: unnormalised numbers invite the reader to compare a student
against perfection rather than against the teacher it was distilled from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lexi_research.format import BandConfig, ValidationError, default_config_path, validate_output
from lexi_research.format.bands import MAX_BAND, MIN_BAND

from .calibration import reliability
from .correction import correction_scores, other_rate, tag_distribution
from .metrics import meaning_metrics
from .report import Report

#: Fields a predictions row must carry. `predict` writes them; `score` refuses
#: anything else rather than silently scoring a partially-parsed file.
REQUIRED_FIELDS = ("req_uid", "text", "gold", "prediction")


class HarnessError(ValueError):
    """The predictions, the ceiling, or the band config made scoring impossible."""


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise HarnessError(f"predictions line {number} is missing {missing}")
            rows.append(row)
    if not rows:
        raise HarnessError(f"{path} holds no predictions")
    return rows


def write_predictions(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return out


def load_ceiling(path: str | Path | None) -> dict[str, Any]:
    """Teacher self-consistency: the highest score the data can support.

    Absent, this raises. Reporting a student's QWK with no ceiling next to it
    invites the reader to compare against 1.0, which no run distilled from this
    teacher could reach.
    """
    if path is None:
        raise HarnessError(
            "no ceiling given. Every metric is reported as a fraction of teacher "
            "self-consistency, so scoring without it would produce numbers that "
            "read as worse than they are. Pass --ceiling."
        )
    resolved = Path(path)
    if not resolved.exists():
        raise HarnessError(f"ceiling artifact {resolved} does not exist")
    payload: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    missing = [key for key in ("meaning_qwk", "correction_edit_f1") if key not in payload]
    if missing:
        raise HarnessError(f"ceiling artifact {resolved} is missing {missing}")
    return payload


def require_calibrated(config: BandConfig) -> None:
    if not config.calibrated:
        raise HarnessError(
            "band_config.json has calibrated: false, so grammar and naturalness "
            "are derived from design guesses. Run `lexi data calibrate` first."
        )


def _valid(row: Mapping[str, Any], config: BandConfig) -> bool:
    prediction = row["prediction"]
    if not isinstance(prediction, Mapping):
        return False
    result = validate_output(dict(prediction), str(row["text"]), config)
    return not isinstance(result, ValidationError)


def format_metrics(rows: Sequence[Mapping[str, Any]], config: BandConfig) -> dict[str, Any]:
    """Whether the student produced something the product could consume at all."""
    parsed = sum(1 for row in rows if isinstance(row.get("prediction"), Mapping))
    valid = sum(1 for row in rows if _valid(row, config))
    retries = sum(int(row.get("retries", 0)) for row in rows)
    identity = 0
    for row in rows:
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            continue
        correction = prediction.get("correction")
        if correction is None:
            identity += 1
            continue
        from lexi_research.format.parser import ParseError, parse_correction

        result = parse_correction(str(correction))
        if not isinstance(result, ParseError) and result.text == str(row["text"]):
            identity += 1
    total = len(rows)
    return {
        "json_parse_rate": parsed / total,
        "validity_rate": valid / total,
        "strip_identity_rate": identity / total,
        "retries_per_row": retries / total,
    }


def _band_pairs(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    predicted, gold = [], []
    for row in rows:
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            continue
        value = prediction.get("meaning")
        reference = row["gold"].get("meaning")
        if not isinstance(value, int) or not isinstance(reference, int):
            continue
        if not (MIN_BAND <= value <= MAX_BAND and MIN_BAND <= reference <= MAX_BAND):
            continue
        predicted.append(value)
        gold.append(reference)
    return predicted, gold


def per_band(predicted: Sequence[int], gold: Sequence[int]) -> dict[str, Any]:
    """Exact and within-1 for each band separately.

    An aggregate hides the band that matters: a student can score well overall
    while being useless at band 2, which is where real learner sentences cluster.
    """
    out: dict[str, Any] = {}
    for band in range(MIN_BAND, MAX_BAND + 1):
        indices = [index for index, value in enumerate(gold) if value == band]
        if not indices:
            out[str(band)] = {"n": 0, "exact": None, "within_one": None}
            continue
        exact = sum(predicted[index] == band for index in indices)
        within = sum(abs(predicted[index] - band) <= 1 for index in indices)
        out[str(band)] = {
            "n": len(indices),
            "exact": exact / len(indices),
            "within_one": within / len(indices),
        }
    return out


def chrf(prediction: str, reference: str, order: int = 6) -> float:
    """Character n-gram F-score. A weak proxy, tagged as one wherever it appears."""
    scores = []
    for n in range(1, order + 1):
        pred_grams = [prediction[i : i + n] for i in range(len(prediction) - n + 1)]
        ref_grams = [reference[i : i + n] for i in range(len(reference) - n + 1)]
        if not pred_grams or not ref_grams:
            continue
        overlap = 0
        remaining = list(ref_grams)
        for gram in pred_grams:
            if gram in remaining:
                remaining.remove(gram)
                overlap += 1
        precision = overlap / len(pred_grams)
        recall = overlap / len(ref_grams)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def feedback_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No hard metric exists here, and the report says so rather than implying one."""
    scores = []
    for row in rows:
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            continue
        predicted = prediction.get("feedback")
        reference = row["gold"].get("feedback")
        if isinstance(predicted, str) and isinstance(reference, str):
            scores.append(chrf(predicted, reference))
    return {"chrf": sum(scores) / len(scores) if scores else 0.0, "n": len(scores)}


def score(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    split: str,
    lineage: Mapping[str, Any],
    ceiling: Mapping[str, Any],
    band_config: BandConfig,
    calibration_bins: int = 10,
) -> Report:
    """Every metric in the design, normalised against the ceiling."""
    require_calibrated(band_config)

    report = Report(
        stage=stage, split=split, rows=len(rows), lineage=lineage, ceiling=dict(ceiling)
    )

    predicted_bands, gold_bands = _band_pairs(rows)
    if not predicted_bands:
        raise HarnessError("no row carried a usable predicted meaning band")
    report.add_group("meaning", meaning_metrics(predicted_bands, gold_bands))
    report.groups["meaning"]["per_band"] = per_band(predicted_bands, gold_bands)

    predictions = [
        row["prediction"].get("correction") if isinstance(row.get("prediction"), Mapping) else None
        for row in rows
    ]
    gold = [row["gold"].get("correction") for row in rows]
    report.add_group("correction", correction_scores(predictions, gold).as_dict(band_config))

    report.add_group("format", format_metrics(rows, band_config))
    report.add_group("feedback", feedback_metrics(rows))

    student_tags = tag_distribution(predictions)
    teacher_tags = tag_distribution(gold)
    report.add_group(
        "taxonomy",
        {
            "other_rate": other_rate(student_tags),
            "teacher_other_rate": other_rate(teacher_tags),
            "student_tags": student_tags,
            "teacher_tags": teacher_tags,
        },
    )

    confidences = [
        float(row["meaning_confidence"])
        for row in rows
        if isinstance(row.get("meaning_confidence"), (int, float))
    ]
    if len(confidences) == len(predicted_bands) and confidences:
        correct = [a == b for a, b in zip(predicted_bands, gold_bands, strict=True)]
        report.groups["calibration"] = reliability(confidences, correct, bins=calibration_bins)
    else:
        report.notes.append(
            "calibration skipped: predictions carry no per-row meaning confidence, "
            "so ECE and the reliability diagram cannot be computed"
        )

    report.notes.append(
        "feedback has no verifiable ground truth; chrF and any judge win-rate are "
        "tagged weak and must be printed with that tag"
    )
    return report


def score_file(
    predictions: str | Path,
    *,
    stage: str,
    split: str,
    lineage: Mapping[str, Any],
    ceiling_path: str | Path | None,
    band_config_path: str | Path | None = None,
    calibration_bins: int = 10,
) -> Report:
    return score(
        read_predictions(predictions),
        stage=stage,
        split=split,
        lineage=lineage,
        ceiling=load_ceiling(ceiling_path),
        band_config=BandConfig.from_json(band_config_path or default_config_path()),
        calibration_bins=calibration_bins,
    )


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Dataset rows from parquet or JSONL, for the predict side."""
    resolved = Path(path)
    if resolved.suffix == ".jsonl":
        with resolved.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    import pyarrow.parquet as pq

    yield from pq.read_table(resolved).to_pylist()


__all__ = [
    "REQUIRED_FIELDS",
    "HarnessError",
    "chrf",
    "feedback_metrics",
    "format_metrics",
    "iter_rows",
    "load_ceiling",
    "per_band",
    "read_predictions",
    "require_calibrated",
    "score",
    "score_file",
    "write_predictions",
]
