"""Tests for the resume log.

The property under test is the one the generation budget depends on: whatever is
in the file is paid for and must not be paid for again, and a kill mid-write must
not make the whole log unreadable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexi_research.data.jsonl_store import ID_FIELD, JsonlStore


class TestAppend:
    def test_round_trips_a_record(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path / "log.jsonl")
        store.append({ID_FIELD: "a", "text": "hello"})
        assert [record["text"] for record in store.read()] == ["hello"]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path / "deep" / "nested" / "log.jsonl")
        store.append({ID_FIELD: "a"})
        assert store.exists()

    def test_rejects_a_record_without_an_id(self, tmp_path: Path) -> None:
        """A record with no id is invisible to resume — better to fail loudly."""
        store = JsonlStore(tmp_path / "log.jsonl")
        with pytest.raises(ValueError, match=ID_FIELD):
            store.append({"text": "no id"})

    def test_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        JsonlStore(path).append({ID_FIELD: "a"})
        JsonlStore(path).append({ID_FIELD: "b"})
        assert JsonlStore(path).completed_ids() == {"a", "b"}

    def test_survives_a_process_boundary(self, tmp_path: Path) -> None:
        """The point of the fsync: a fresh reader sees everything written."""
        path = tmp_path / "log.jsonl"
        writer = JsonlStore(path)
        for index in range(20):
            writer.append({ID_FIELD: f"r{index}"})
        assert len(JsonlStore(path).completed_ids()) == 20


class TestResume:
    def test_completed_ids_is_empty_before_any_write(self, tmp_path: Path) -> None:
        assert JsonlStore(tmp_path / "missing.jsonl").completed_ids() == set()

    def test_read_of_a_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        assert list(JsonlStore(tmp_path / "missing.jsonl").read()) == []

    def test_torn_final_line_is_skipped(self, tmp_path: Path) -> None:
        """A kill mid-write leaves a partial line; it must cost one record, not all."""
        path = tmp_path / "log.jsonl"
        store = JsonlStore(path)
        store.append({ID_FIELD: "a"})
        store.append({ID_FIELD: "b"})
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"req_uid": "c", "tex')

        assert JsonlStore(path).completed_ids() == {"a", "b"}

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        path.write_text('{"req_uid": "a"}\n\n\n{"req_uid": "b"}\n', encoding="utf-8")
        assert JsonlStore(path).completed_ids() == {"a", "b"}

    def test_non_dict_json_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        path.write_text('[1, 2]\n"a string"\n{"req_uid": "a"}\n', encoding="utf-8")
        assert JsonlStore(path).completed_ids() == {"a"}

    def test_non_string_ids_do_not_count_as_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        path.write_text('{"req_uid": 7}\n{"req_uid": "a"}\n', encoding="utf-8")
        assert JsonlStore(path).completed_ids() == {"a"}


class TestLatestById:
    def test_last_write_wins(self, tmp_path: Path) -> None:
        """A repaired record must supersede the one it replaces."""
        store = JsonlStore(tmp_path / "log.jsonl")
        store.append({ID_FIELD: "a", "text": "first"})
        store.append({ID_FIELD: "a", "text": "second"})
        assert store.latest_by_id()["a"]["text"] == "second"

    def test_read_still_yields_both_copies(self, tmp_path: Path) -> None:
        """Dedup is the caller's policy, so the raw log keeps the history."""
        store = JsonlStore(tmp_path / "log.jsonl")
        store.append({ID_FIELD: "a", "text": "first"})
        store.append({ID_FIELD: "a", "text": "second"})
        assert len(list(store.read())) == 2


class TestEncoding:
    def test_non_ascii_survives(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path / "log.jsonl")
        store.append({ID_FIELD: "a", "text": "cà phê sữa đá"})
        assert next(iter(store.read()))["text"] == "cà phê sữa đá"

    def test_records_are_written_with_sorted_keys(self, tmp_path: Path) -> None:
        """Deterministic serialisation: the log itself is diffable across runs."""
        path = tmp_path / "log.jsonl"
        JsonlStore(path).append({"z": 1, ID_FIELD: "a", "b": 2})
        line = path.read_text(encoding="utf-8").strip()
        assert list(json.loads(line)) == sorted(json.loads(line))

    def test_one_record_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        store = JsonlStore(path)
        store.append({ID_FIELD: "a", "text": "line\nbreak"})
        store.append({ID_FIELD: "b"})
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2
