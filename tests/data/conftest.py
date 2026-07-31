"""A synthetic Cambridge-shaped SQLite fixture, built in-process.

CI never sees the real source: it is 150MB of licensed dictionary text that this
repo has no redistribution rights to. The fixture is generated instead of
committed so the schema stays visibly in sync with the real one — the DDL below
is copied from `/home/qninh/projects/lexi-ai/data` and a drifting source shows up
as a failing export rather than as a stale binary nobody re-reads.

The rows deliberately include the source's defects: a dirty POS (`adj`, `V`,
`''`), non-lexical entry types (`suffix`, `symbol`), an empty definition, a
duplicate sense, a multiword headword, and a bare `sb` placeholder. Every
quarantine reason the export can emit has a row here that triggers it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

#: Copied verbatim from the real source's `sqlite_master`. Only the three tables
#: the export reads are recreated; the other eight are irrelevant to it.
_DDL = """
CREATE TABLE words (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    word         TEXT    UNIQUE NOT NULL,
    display_form TEXT,
    entry_type   TEXT    DEFAULT 'word',
    status       TEXT    DEFAULT 'pending',
    crawled_at   TEXT,
    error_msg    TEXT
);
CREATE TABLE entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id         INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    entry_order     INTEGER DEFAULT 0,
    headword        TEXT,
    pos             TEXT,
    grammar         TEXT,
    pronunciation_uk TEXT,
    pronunciation_us TEXT,
    audio_uk_url    TEXT,
    audio_us_url    TEXT,
    dictionary_source TEXT
);
CREATE TABLE senses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    sense_order INTEGER DEFAULT 0,
    guideword   TEXT,
    definition  TEXT NOT NULL,
    cefr_level  TEXT,
    grammar     TEXT,
    domain      TEXT,
    labels      TEXT DEFAULT '[]',
    phrase_title TEXT
);
"""

#: `(word_slug, display_form, headword, pos_raw, [(definition, cefr), ...])`.
#: The comment on each row names what it is there to exercise.
_ROWS: tuple[tuple[str, str, str, str, tuple[tuple[str, str | None], ...]], ...] = (
    # Clean multi-sense noun: the ordinary case, and a target with 2 senses so
    # `distinct_targets` differs from the row count.
    ("light", "light", "light", "noun", (
        ("the brightness that comes from the sun, fire, or a lamp", "A1"),
        ("a device that produces brightness", "A2"),
    )),
    # Same headword, different POS: must not collide on `sense_uid`.
    ("light", "light", "light", "adjective", (("not heavy", "A2"),)),
    # Abbreviated POS: `adj` must normalise, not quarantine.
    ("bright", "bright", "bright", "adj", (("full of light", "A2"),)),
    # Single-letter POS.
    ("run", "run", "run", "V", (("to move quickly on foot", "A1"),)),
    # Subtype folded into its head POS.
    ("scissors", "scissors", "scissors", "plural noun", (("a tool for cutting", "A2"),)),
    ("must", "must", "must", "modal verb", (("used to say something is necessary", "A2"),)),
    # Multiword targets: flagged, not dropped.
    ("give-up", "give up", "give up", "phrasal verb", (("to stop doing something", "B1"),)),
    ("piece-of-cake", "piece of cake", "piece of cake", "idiom", (("something very easy", "B2"),)),
    # Bare `sb` placeholder. Brace forms do not occur in this source.
    ("put-sb-down", "put sb down", "put sb down", "phrasal verb", (("to criticise someone", "B2"),)),
    # `something` must NOT match the placeholder regex.
    ("something", "something", "something", "noun", (("an unknown thing", "A1"),)),
    # No CEFR label: most of the source has none.
    ("obfuscate", "obfuscate", "obfuscate", "verb", (("to make something unclear", None),)),
    # Excluded POS, one per exclusion family.
    ("ness", "-ness", "-ness", "suffix", (("used to form nouns", None),)),
    ("percent-sign", "%", "%", "symbol", (("the symbol for percent", None),)),
    ("the", "the", "the", "definite article", (("used before a noun", "A1"),)),
    ("of", "of", "of", "preposition", (("belonging to", "A1"),)),
    ("kg", "kg", "kg", "written abbreviation", (("kilogram", None),)),
    # Empty POS.
    ("mystery-pos", "mystery", "mystery", "", (("something unexplained", None),)),
    # A POS in neither table: the drift signal `unmappable_pos` exists for.
    ("newfangled-pos", "newfangled", "newfangled", "quasi-verb", (("a novel category", None),)),
    # Whitespace-only definition: `empty_definition` (the column is NOT NULL, so
    # the source cannot hold a true null here).
    ("hollow", "hollow", "hollow", "noun", (("   ", None),)),
    # Definition below the minimum length.
    ("tiny-def", "tiny", "tiny", "noun", (("a", None),)),
    # Headword absent: falls back to `display_form`.
    ("fallback", "fallback form", "", "noun", (("relies on display_form", None),)),
    # Case and whitespace variants of one lemma: `target_norm` must fold them
    # into a single split group.
    ("Paris", "Paris", "  Paris  ", "noun", (("the capital of France", "A1"),)),
    ("paris-2", "paris", "paris", "noun", (("a plaster material", "C1"),)),
)

#: Exercised separately: the same (target, pos, definition) twice → `duplicate_uid`.
_DUPLICATE = ("dup", "dup", "dup", "noun", ("a repeated sense", "B1"))


def build_fixture_db(path: Path) -> Path:
    """Write a synthetic Cambridge-shaped database to `path`."""
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_DDL)
        for slug, display, headword, pos, senses in _ROWS:
            word_id = _ensure_word(connection, slug, display)
            entry_id = connection.execute(
                "INSERT INTO entries (word_id, headword, pos) VALUES (?, ?, ?)",
                (word_id, headword, pos),
            ).lastrowid
            for order, (definition, cefr) in enumerate(senses):
                connection.execute(
                    "INSERT INTO senses (entry_id, sense_order, definition, cefr_level)"
                    " VALUES (?, ?, ?, ?)",
                    (entry_id, order, definition, cefr),
                )

        slug, display, headword, pos, (definition, cefr) = _DUPLICATE
        word_id = _ensure_word(connection, slug, display)
        for _ in range(2):
            entry_id = connection.execute(
                "INSERT INTO entries (word_id, headword, pos) VALUES (?, ?, ?)",
                (word_id, headword, pos),
            ).lastrowid
            connection.execute(
                "INSERT INTO senses (entry_id, definition, cefr_level) VALUES (?, ?, ?)",
                (entry_id, definition, cefr),
            )
        connection.commit()
    finally:
        connection.close()
    return path


def _ensure_word(connection: sqlite3.Connection, slug: str, display: str) -> int:
    """Insert a word row, or return the existing id — `words.word` is UNIQUE."""
    row = connection.execute("SELECT id FROM words WHERE word = ?", (slug,)).fetchone()
    if row is not None:
        return int(row[0])
    cursor = connection.execute(
        "INSERT INTO words (word, display_form) VALUES (?, ?)", (slug, display)
    )
    return int(cursor.lastrowid or 0)


@pytest.fixture(scope="session")
def mini_cambridge(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to the synthetic source database."""
    return build_fixture_db(tmp_path_factory.mktemp("source") / "mini_cambridge.sqlite")
