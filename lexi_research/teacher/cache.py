"""Content-addressed response cache — the thing that makes generation resumable.

A bulk run is thousands of paid calls. Without a cache, an interruption at 80%
either re-spends the whole budget or leaves the pipeline in a state nobody can
reason about. Keying on the *request content* rather than on a row index means
resume needs no bookkeeping: rerun the stage, and only the calls whose content is
new actually fire.

The key covers the model, the prompt hash, and the serialized request, so any of
these changing correctly invalidates the entry:

    key = sha256(model | prompt_hash | canonical_json(request))

Storage is JSONL sharded by the key's first two hex characters — append-only
writes survive a kill mid-run, and 256 shards keep any single file scannable.
This is a cache, not an artifact: it is never DVC-tracked. Its hit rate is
reported instead, because that number is what explains a resumed run's spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

#: How many leading hex chars of the key name the shard file.
_SHARD_WIDTH = 2


def cache_key(model: str, prompt_hash: str, request: Any) -> str:
    """Content address for one request.

    `request` is serialized with sorted keys, so a dict that differs only in
    insertion order hits the same entry rather than paying twice for it.
    """
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(f"{model}|{prompt_hash}|{payload}".encode())
    return digest.hexdigest()


class ResponseCache:
    """A sharded JSONL store of `key -> response payload`.

    Lazily loads a shard on first touch and keeps it in memory, which is what
    makes a resumed run's lookups cheap. Thread-safe because the client can run
    many requests concurrently.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._shards: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _shard_path(self, key: str) -> Path:
        return self.root / f"{key[:_SHARD_WIDTH]}.jsonl"

    def _load_shard(self, prefix: str) -> dict[str, Any]:
        """Read one shard file into memory, tolerating a torn final line.

        A run killed mid-write can leave a partial last line. Skipping
        unparseable lines costs at most one cache entry — refusing to load would
        cost the whole shard, which is the opposite of what a resume cache is
        for.
        """
        cached = self._shards.get(prefix)
        if cached is not None:
            return cached

        entries: dict[str, Any] = {}
        path = self.root / f"{prefix}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("key")
                    if isinstance(key, str):
                        # A later record for the same key wins: a repaired entry
                        # should shadow the one it replaces.
                        entries[key] = record.get("response")
        self._shards[prefix] = entries
        return entries

    def get(self, key: str) -> Any | None:
        """Cached response for `key`, or `None` on a miss.

        A cached `null` response is indistinguishable from a miss here. Nothing
        stores one — every cached value is a schema payload dict — so the
        ambiguity costs a repeat call at worst, never a wrong answer.
        """
        with self._lock:
            entries = self._load_shard(key[:_SHARD_WIDTH])
            value = entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            return value

    def put(self, key: str, response: Any) -> None:
        """Append a response, making it visible to this process immediately."""
        with self._lock:
            prefix = key[:_SHARD_WIDTH]
            entries = self._load_shard(prefix)
            entries[key] = response
            self.root.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"key": key, "response": response}, ensure_ascii=False)
            with self._shard_path(key).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                # An interrupted run must not lose the calls it already paid for,
                # so durability here is worth more than write throughput.
                handle.flush()
                os.fsync(handle.fileno())
            self.writes += 1

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 4),
        }


class NullCache(ResponseCache):
    """A cache that never hits — for the parity checks that must re-ask the model.

    Phase 2's `K=6` vs `K=1` comparison and Phase 4's self-consistency re-grade
    both depend on the teacher actually answering again. Sharing the real cache's
    interface keeps that a constructor choice instead of a branch in the client.
    """

    def __init__(self) -> None:
        super().__init__(Path(tempfile.gettempdir()) / "lexi-research-null-cache")

    def get(self, key: str) -> Any | None:  # noqa: ARG002 - signature is the point
        del key
        self.misses += 1
        return None

    def put(self, key: str, response: Any) -> None:  # noqa: ARG002 - signature is the point
        del key, response


__all__ = ["NullCache", "ResponseCache", "cache_key"]
