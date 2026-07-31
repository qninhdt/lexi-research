"""Phase 5: validate teacher labels, balance them, and create group-safe splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from lexi_research.format import BandConfig, ValidationError, validate_output

from .balance import balance_rows, distribution
from .split import split_rows, strict_test_rows


def process_rows(rows: Sequence[dict[str, Any]], config: BandConfig, *, seed: int, version: str, max_stratum_share: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for row in rows:
        checked = validate_output({key: row.get(key) for key in ("correction", "meaning", "feedback")}, str(row.get("text", "")), config)
        if isinstance(checked, ValidationError):
            rejects.append({**row, "reject_reason": checked.code, "reject_detail": checked.detail})
            continue
        edits = checked.edits or []
        clean.append({**row, "meaning": checked.meaning, "feedback": checked.feedback, "grammar": checked.bands.grammar, "naturalness": checked.bands.naturalness, "tags": sorted({edit.tag for edit in edits}), "n_edits": len(edits), "n_words": len(str(row["text"]).split())})
    balanced = balance_rows(clean, max_stratum_share=max_stratum_share, seed=seed)
    split = split_rows(balanced, seed=seed, version=version)
    report = {"input_rows": len(rows), "clean_rows": len(clean), "rejected_rows": len(rejects), "reject_reasons": dict(Counter(row["reject_reason"] for row in rejects)), "before_balance": distribution(clean), "after_balance": distribution(balanced), "contamination": split.contamination, "strict_test_rows": len(strict_test_rows(split.rows))}
    return list(split.rows), rejects, report


def process_parquet(texts_path: str | Path, labels_path: str | Path, out_dir: str | Path, config: BandConfig, *, seed: int, version: str, max_stratum_share: float) -> dict[str, Any]:
    texts = {row["req_uid"]: row for row in pq.read_table(texts_path).to_pylist()}
    rows = [{**texts[row["req_uid"]], **row} for row in pq.read_table(labels_path).to_pylist() if row["req_uid"] in texts]
    processed, rejects, report = process_rows(rows, config, seed=seed, version=version, max_stratum_share=max_stratum_share)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(processed), out / "processed.parquet")
    pq.write_table(pa.Table.from_pylist(rejects), out / "rejects.parquet")
    for name in ("train", "val", "test"):
        pq.write_table(pa.Table.from_pylist([row for row in processed if row["split"] == name]), out / f"{name}.parquet")
    pq.write_table(pa.Table.from_pylist(strict_test_rows(processed)), out / "test_strict.parquet")
    report["band_config_sha256"] = hashlib.sha256(json.dumps(config.__dict__, default=str, sort_keys=True).encode()).hexdigest()
    (out / "data-quality.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


__all__ = ["process_parquet", "process_rows"]
