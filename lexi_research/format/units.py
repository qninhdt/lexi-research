"""Unit lexer and character offset mapping for numbered English sentences.

Splits an English sentence into 1-based indexed units (words, contractions, numbers,
and individual punctuation marks) while preserving precise start and end character
offsets into the raw original string.
"""

from __future__ import annotations

import re
from typing import NamedTuple

__all__ = [
    "Unit",
    "SpanEdit",
    "UNIT_RE",
    "lex_units",
    "format_numbered_input",
]


class Unit(NamedTuple):
    """A 1-based indexed word or punctuation unit with raw character offsets."""

    index: int  # 1-based unit index
    token: str  # text of the unit
    start_char: int  # character offset in original raw sentence (inclusive)
    end_char: int  # character offset in original raw sentence (exclusive)


class SpanEdit(NamedTuple):
    """A 1-based span edit over word units."""

    start: int  # 1-based unit index (inclusive)
    end: int  # 1-based unit index (exclusive). start == end indicates insertion.
    tag: str  # taxonomy tag
    replacement: str  # replacement text (empty string for deletion)


#: Regex that splits sentences into units:
#: 1. Negative contractions prefix: e.g. 'could' in 'couldn't'
#: 2. Contraction suffix: 'n't'
#: 3. Alphanumeric words (Unicode-aware): e.g. 'café', 'sofà', '1990'
#: 4. Individual punctuation symbols: e.g. '.', ',', '?', '!', ':', ';', '-', etc.
UNIT_RE = re.compile(r"[^\W\d_]+?(?=n[\x27\x27]t\b)|n[\x27\x27]t|\w+|[^\w\s]")


def lex_units(text: str) -> list[Unit]:
    """Tokenize a sentence into a list of 1-based indexed units with char offsets."""
    return [
        Unit(
            index=idx,
            token=match.group(0),
            start_char=match.start(),
            end_char=match.end(),
        )
        for idx, match in enumerate(UNIT_RE.finditer(text), start=1)
    ]


def format_numbered_input(text: str) -> str:
    """Format a raw sentence into newline-separated numbered units: '1 He\\n2 speak\\n3 English'."""
    units = lex_units(text)
    return "\n".join(f"{u.index} {u.token}" for u in units)
