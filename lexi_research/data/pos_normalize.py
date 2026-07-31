"""Normalize the deliberately messy Cambridge entry POS field."""

from __future__ import annotations

import re

_NORMALIZED: dict[str, str] = {
    "n": "noun",
    "noun": "noun",
    "plural noun": "noun",
    "verb": "verb",
    "v": "verb",
    "auxiliary verb": "verb",
    "modal verb": "verb",
    "adjective": "adjective",
    "adj": "adjective",
    "adverb": "adverb",
    "idiom": "idiom",
    "phrasal verb": "phrasal verb",
    "collocation": "collocation",
    "phrase": "phrase",
    "exclamation": "exclamation",
    "interjection": "exclamation",
    "adjective, adverb": "adjective",
    "adverb, adjective": "adverb",
    "noun or exclamation": "noun",
}
_EXCLUDED = frozenset(
    {
        "",
        "suffix",
        "prefix",
        "combining form",
        "abbreviation",
        "written abbreviation",
        "symbol",
        "number",
        "ordinal number",
        "modifier",
        "predeterminer",
        "indefinite article",
        "definite article",
        # Function-word entries are dictionary data, but not viable targets for
        # this task: learners cannot be asked to demonstrate one specific sense
        # of "of", "the", or a pronoun in the same way as a lexical item.
        "preposition",
        "pronoun",
        "conjunction",
        "determiner",
        "trademark",
        "short form",
    }
)
_MULTIWORD_POS = frozenset({"idiom", "phrasal verb", "collocation", "phrase"})


def normalize_pos(value: str | None) -> str | None:
    """Return a canonical lexical POS, or ``None`` for an explicit exclusion."""
    raw = (value or "").strip().lower()
    if raw in _EXCLUDED:
        return None
    return _NORMALIZED.get(raw)


def is_explicitly_excluded(value: str | None) -> bool:
    return (value or "").strip().lower() in _EXCLUDED


def is_lexical(value: str | None) -> bool:
    return normalize_pos(value) is not None


def normalize_target(value: str) -> str:
    """Stable grouping form that leaves display text untouched in the artifact."""
    return re.sub(r"\s+", " ", value.strip().casefold())


def is_multiword(target: str, pos: str) -> bool:
    return pos in _MULTIWORD_POS or " " in target.strip()


def has_placeholder(target: str) -> bool:
    return bool(re.search(r"(?<!\w)(?:sb|sth)(?!\w)", target, flags=re.IGNORECASE))
