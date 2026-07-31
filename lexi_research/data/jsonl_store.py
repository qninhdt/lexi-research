"""Append-only JSONL store keyed by request id — how a generation run resumes.

A bulk run is thousands of paid calls spread over hours. The failure to design
against is not a crash; it is a crash that leaves the pipeline unable to say
which calls were already paid for. Writing each record as one line, flushed and
fsynced, keyed by a `req_uid` derived from the request content, makes that
question answerable by reading the file: whatever is present is done, whatever is
absent is not.

Parquet is the artifact; this is the write-ahead log in front of it. The two are
separate on purpose — Parquet cannot be appended to safely mid-run, and a
partially written Parquet file is unreadable rather than short.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: The key every record must carry. Content-derived upstream, so a re-run
#: produces the same id for the same request and the skip logic needs no
#: separate bookkeeping.
ID_FIELD = "req_uid"


class JsonlStore:
    """A resumable append-only record log.

    `completed_ids()` is the resume primitive: read it before a run and skip the
    work it already covers. Torn final lines — the normal result of a kill
    mid-write — are skipped rather than fatal, costing at most the one record
    that was in flight.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.appended = 0

    def exists(self) -> bool:
        return self.path.exists()

    def completed_ids(self) -> set[str]:
        """Ids already durably recorded. Empty when the log does not exist yet."""
        return {
            record[ID_FIELD]
            for record in self.read()
            if isinstance(record.get(ID_FIELD), str)
        }

    def read(self) -> Iterator[dict[str, Any]]:
        """Every parseable record, in write order.

        A record written twice yields twice: dedup is the caller's business,
        because "last write wins" and "first write wins" are both legitimate
        depending on whether a repair overwrote a bad row.
        """
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A partial trailing line from an interrupted write. Only the
                    # last line can be torn, but skipping any unparseable line is
                    # strictly safer than refusing to load the whole log.
                    continue
                if isinstance(record, dict):
                    yield record

    def append(self, record: dict[str, Any]) -> None:
        """Durably append one record.

        `fsync` per record is deliberate: the point of this file is that a
        killed run does not re-spend money, and an unflushed buffer defeats that
        entirely. At a few thousand records the cost is irrelevant next to the
        latency of the API calls it is protecting.
        """
        if ID_FIELD not in record:
            raise ValueError(f"record has no {ID_FIELD!r}: {sorted(record)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.appended += 1

    def latest_by_id(self) -> dict[str, dict[str, Any]]:
        """Records keyed by id, last write winning.

        Last-write-wins is the right rule for converting a log to an artifact: a
        record appended later is either a retry that succeeded or a repair, and
        in both cases it supersedes what came before.
        """
        return {record[ID_FIELD]: record for record in self.read() if ID_FIELD in record}


__all__ = ["ID_FIELD", "JsonlStore"]
