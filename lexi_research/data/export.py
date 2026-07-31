"""Read-only Cambridge SQLite export with deterministic provenance and quarantine."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .pos_normalize import (
    has_placeholder,
    is_explicitly_excluded,
    is_multiword,
    normalize_pos,
    normalize_target,
)


@dataclass(frozen=True)
class ExportSummary:
    source_db_sha256: str
    pool_rows: int
    target_groups: int
    quarantine_rows: int


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_parquet(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    bool_columns = {"is_multiword", "is_placeholder"}
    schema = pa.schema(
        [(column, pa.bool_() if column in bool_columns else pa.string()) for column in columns]
    )
    table = pa.Table.from_pydict(
        {column: [row.get(column) for row in rows] for column in columns}, schema=schema
    )
    pq.write_table(table, path, compression="zstd", use_dictionary=False, write_statistics=False)


def export_senses(source: str | Path, out_dir: str | Path, *, min_definition_chars: int = 3) -> ExportSummary:
    """Export each unique, lexical sense and quarantine every omitted source row."""
    source_path = Path(source)
    if not source_path.is_file():
        raise ValueError(f"source SQLite file does not exist: {source_path}")
    if min_definition_chars < 1:
        raise ValueError("min_definition_chars must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = fingerprint_file(source_path)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT s.id AS source_sense_id, s.definition, s.cefr_level,
                   e.headword, e.pos
            FROM senses AS s
            JOIN entries AS e ON e.id = s.entry_id
            ORDER BY s.id
            """
        )
        for row in rows:
            source_id = str(row["source_sense_id"])
            definition = (row["definition"] or "").strip()
            target = (row["headword"] or "").strip()
            raw_pos = row["pos"]
            reason: str | None = None
            if len(definition) < min_definition_chars:
                reason = "empty_definition" if not definition else "definition_too_short"
            elif not target:
                reason = "empty_target"
            else:
                pos = normalize_pos(raw_pos)
                if pos is None:
                    reason = "excluded_pos" if is_explicitly_excluded(raw_pos) else "unmappable_pos"
                else:
                    target_norm = normalize_target(target)
                    identity = f"{target_norm}|{pos}|{definition}"
                    sense_uid = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                    if sense_uid in seen:
                        reason = "duplicate_uid"
                    else:
                        seen.add(sense_uid)
                        kept.append(
                            {
                                "sense_uid": sense_uid,
                                "source_sense_id": source_id,
                                "target": target,
                                "target_norm": target_norm,
                                "pos": pos,
                                "definition": definition,
                                "cefr": row["cefr_level"] or None,
                                "is_multiword": is_multiword(target, pos),
                                "is_placeholder": has_placeholder(target),
                                "source_db_sha256": source_hash,
                            }
                        )
            if reason is not None:
                rejected.append({"source_sense_id": source_id, "reason": reason, "raw_pos": raw_pos or ""})
    finally:
        conn.close()
    kept.sort(key=lambda row: row["sense_uid"])
    rejected.sort(key=lambda row: (row["reason"], row["source_sense_id"]))
    _write_parquet(kept, out / "senses_pool.parquet", list(kept[0]) if kept else ["sense_uid"])
    _write_parquet(rejected, out / "quarantine.parquet", ["source_sense_id", "reason", "raw_pos"])
    quality = {
        "pool_rows": len(kept),
        "target_groups": len({row["target_norm"] for row in kept}),
        "pos_counts": dict(sorted(Counter(row["pos"] for row in kept).items())),
        "cefr_counts": dict(sorted(Counter(row["cefr"] or "null" for row in kept).items())),
        "multiword_rows": sum(row["is_multiword"] for row in kept),
        "placeholder_rows": sum(row["is_placeholder"] for row in kept),
        "quarantine_counts": dict(sorted(Counter(row["reason"] for row in rejected).items())),
    }
    (out / "data-quality.json").write_text(json.dumps(quality, sort_keys=True, indent=2) + "\n")
    manifest = {"source_db_sha256": source_hash, "source_file": source_path.name, **asdict(ExportSummary(source_hash, len(kept), quality["target_groups"], len(rejected)))}
    (out / "source-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return ExportSummary(source_hash, len(kept), quality["target_groups"], len(rejected))
