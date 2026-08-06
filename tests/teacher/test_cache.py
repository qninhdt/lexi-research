"""The resume cache: keying, durability, and the torn-write tolerance."""

from __future__ import annotations

import json
from pathlib import Path

from lexi_research.teacher import NullCache, ResponseCache, cache_key


def test_key_is_stable_across_dict_order() -> None:
    """A request that differs only in key order must not pay twice."""
    left = cache_key("m", "h", {"a": 1, "b": 2})
    right = cache_key("m", "h", {"b": 2, "a": 1})
    assert left == right


def test_key_changes_with_model_prompt_and_request() -> None:
    base = cache_key("m", "h", {"a": 1})
    assert cache_key("other", "h", {"a": 1}) != base
    assert cache_key("m", "other", {"a": 1}) != base
    assert cache_key("m", "h", {"a": 2}) != base


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("abc123", {"meaning": 4})
    assert cache.get("abc123") == {"meaning": 4}
    assert cache.hits == 1


def test_get_miss_returns_none(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    assert cache.get("deadbeef") is None
    assert cache.misses == 1


def test_entries_survive_a_new_process(tmp_path: Path) -> None:
    """Resume depends on this: a fresh instance sees what the last run wrote."""
    ResponseCache(tmp_path).put("ab" + "0" * 8, {"meaning": 2})
    assert ResponseCache(tmp_path).get("ab" + "0" * 8) == {"meaning": 2}


def test_keys_are_sharded_by_prefix(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("ab1", {"v": 1})
    cache.put("cd2", {"v": 2})
    assert {path.name for path in tmp_path.glob("*.jsonl")} == {"ab.jsonl", "cd.jsonl"}


def test_torn_final_line_costs_one_entry_not_the_shard(tmp_path: Path) -> None:
    """A run killed mid-write leaves a partial line; the rest must still load."""
    shard = tmp_path / "ab.jsonl"
    good = json.dumps({"key": "ab_good", "response": {"v": 1}})
    shard.write_text(good + '\n{"key": "ab_torn", "resp', encoding="utf-8")

    cache = ResponseCache(tmp_path)
    assert cache.get("ab_good") == {"v": 1}
    assert cache.get("ab_torn") is None


def test_later_record_shadows_an_earlier_one(tmp_path: Path) -> None:
    """Append-only storage means a repair is a later line for the same key."""
    shard = tmp_path / "ab.jsonl"
    shard.write_text(
        json.dumps({"key": "ab_k", "response": {"v": 1}})
        + "\n"
        + json.dumps({"key": "ab_k", "response": {"v": 2}})
        + "\n",
        encoding="utf-8",
    )
    assert ResponseCache(tmp_path).get("ab_k") == {"v": 2}


def test_stats_report_hit_rate(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("ab1", {"v": 1})
    cache.get("ab1")
    cache.get("ab2")
    assert cache.stats() == {
        "lookups": 2,
        "hits": 1,
        "misses": 1,
        "writes": 1,
        "hit_rate": 0.5,
    }


def test_delete_invalidates_an_entry_across_processes(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put("ab1", {"v": 1})
    cache.delete("ab1")

    assert cache.get("ab1") is None
    assert ResponseCache(tmp_path).get("ab1") is None


def test_null_cache_never_hits_and_writes_nothing(tmp_path: Path) -> None:
    """The parity checks need the teacher to actually answer again."""
    cache = NullCache()
    cache.put("ab1", {"v": 1})
    assert cache.get("ab1") is None
    assert not list(tmp_path.glob("*.jsonl"))
