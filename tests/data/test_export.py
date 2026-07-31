import json
import sqlite3

import pyarrow.parquet as pq

from lexi_research.data.export import export_senses


def _fixture(path):
    db = sqlite3.connect(path)
    db.executescript("""
      create table words (id integer primary key, word text, display_form text);
      create table entries (id integer primary key, word_id integer, headword text, pos text);
      create table senses (id integer primary key, entry_id integer, definition text, cefr_level text);
    """)
    db.executemany("insert into words values (?, ?, ?)", [(1, "bright", "bright"), (2, "put-sb-down", "put sb down")])
    db.executemany("insert into entries values (?, ?, ?, ?)", [(1, 1, "bright", "adj"), (2, 1, "bright", "adjective"), (3, 2, "put sb down", "phrasal verb"), (4, 1, "x", "suffix")])
    db.executemany("insert into senses values (?, ?, ?, ?)", [(1, 1, "full of light", "A2"), (2, 2, "full of light", "A2"), (3, 3, "to criticize", "B2"), (4, 4, "a suffix", None)])
    db.commit(); db.close()


def test_export_is_deterministic_and_quarantines(tmp_path) -> None:
    source = tmp_path / "source.sqlite"; _fixture(source)
    one, two = tmp_path / "one", tmp_path / "two"
    result_one = export_senses(source, one)
    result_two = export_senses(source, two)
    assert result_one == result_two
    assert (one / "senses_pool.parquet").read_bytes() == (two / "senses_pool.parquet").read_bytes()
    rows = pq.read_table(one / "senses_pool.parquet").to_pylist()
    assert len(rows) == 2
    assert any(row["is_placeholder"] for row in rows)
    quality = json.loads((one / "data-quality.json").read_text())
    assert quality["quarantine_counts"] == {"duplicate_uid": 1, "excluded_pos": 1}
