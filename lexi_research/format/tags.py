"""The closed error taxonomy: 16 tags and the two groups bands are derived from.

The *set* of legal tags lives in code, not in `band_config.json`. The config file
supplies numeric weights and thresholds — data that gets recalibrated — while the
taxonomy itself is a contract shared by the prompt, the teacher, the student, and
the validator. Keeping it here means a config edit cannot silently widen the set
of tags the model is allowed to emit.

There is deliberately no tag for wrong meaning: wrong meaning does not live in a
span and cannot be fixed by a replacement, so it is carried by the `meaning` band
instead.
"""

from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    """A string enum that actually behaves like its value, back-ported from 3.11.

    `str` mixin alone is not enough, and both gaps fail *silently* rather than
    raising, which is why they are closed here once instead of per enum:

    - `Enum.__str__` renders `Tag.AGR`, so `str(tag)` and f-strings would yield
      a string that is not a legal tag — and tags are looked up by string.
    - `Enum.__hash__` hashes the *name*, so `Tag.AGR in {"agr"}` is False even
      though `Tag.AGR == "agr"` is True, making set and dict lookups miss.
    """

    __str__ = str.__str__
    __format__ = str.__format__  # type: ignore[assignment]
    __hash__ = str.__hash__


class Tag(_StrEnum):
    """One of the 16 legal edit tags."""

    # Correctness — mechanical / grammatical
    PUNC = "punc"
    SP = "sp"
    ART = "art"
    NUM = "num"
    POSS = "poss"
    PREP = "prep"
    PART = "part"
    AGR = "agr"
    TENSE = "tense"
    FORM = "form"
    PRON = "pron"
    ORDER = "order"
    # Usage — idiomaticity / phrasing
    COLL = "coll"
    WORD = "word"
    UNNAT = "unnat"
    # Catch-all. Exists to measure what the taxonomy is missing: without it the
    # teacher crams misfits into the nearest tag and the noise becomes invisible.
    OTHER = "other"


class TagGroup(_StrEnum):
    """The two penalty groups. `grammar` comes from one, `naturalness` from the other."""

    CORRECTNESS = "correctness"
    USAGE = "usage"


# `other` counts as correctness: a catch-all error is more likely grammatical
# than stylistic.
GROUP_OF: dict[Tag, TagGroup] = {
    Tag.PUNC: TagGroup.CORRECTNESS,
    Tag.SP: TagGroup.CORRECTNESS,
    Tag.ART: TagGroup.CORRECTNESS,
    Tag.NUM: TagGroup.CORRECTNESS,
    Tag.POSS: TagGroup.CORRECTNESS,
    Tag.PREP: TagGroup.CORRECTNESS,
    Tag.PART: TagGroup.CORRECTNESS,
    Tag.AGR: TagGroup.CORRECTNESS,
    Tag.TENSE: TagGroup.CORRECTNESS,
    Tag.FORM: TagGroup.CORRECTNESS,
    Tag.PRON: TagGroup.CORRECTNESS,
    Tag.ORDER: TagGroup.CORRECTNESS,
    Tag.OTHER: TagGroup.CORRECTNESS,
    Tag.COLL: TagGroup.USAGE,
    Tag.WORD: TagGroup.USAGE,
    Tag.UNNAT: TagGroup.USAGE,
}

#: Every legal tag as a plain string — the closed set the validator checks against.
TAGS: frozenset[str] = frozenset(tag.value for tag in Tag)

#: Tag pairs a grader routinely confuses. They MUST carry equal weight, so that
#: confusing one for the other cannot move a band. This is what makes deriving
#: bands from noisy tags viable, and it is asserted by a test.
CONFUSABLE_PAIRS: tuple[tuple[Tag, Tag], ...] = (
    (Tag.WORD, Tag.COLL),
    (Tag.PREP, Tag.PART),
)


def group_of(tag: str) -> TagGroup:
    """Group for a tag given as a plain string.

    Raises `KeyError` for an unknown tag — callers grade model output through
    `validate_output` first, which rejects tags outside `TAGS`.
    """
    return GROUP_OF[Tag(tag)]
