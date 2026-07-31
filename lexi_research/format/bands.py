"""Derive the `grammar` and `naturalness` bands from parsed edits.

The model never emits these two bands. Code derives them from the correction's
error tags, which buys three things: the same error set always scores the same
way (a model emitting bands disagrees with itself), the anchor is an explicit
weight table rather than a vague rubric, and the cut points can be retuned
without retraining.

Weights and thresholds are *data* (`band_config.json`) rather than constants,
and the config ships with the checkpoint — an adapter without its config
produces meaningless bands.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parser import Edit
from .tags import TAGS, TagGroup

#: Bands run 0..4 inclusive, for both `meaning` and the derived pair.
MIN_BAND = 0
MAX_BAND = 4


def _positive_version(value: Any) -> int:
    version = int(value)
    if version < 1:
        raise ValueError("band config version must be positive")
    return version


@dataclass(frozen=True)
class Bands:
    """The two derived bands."""

    grammar: int
    naturalness: int


@dataclass(frozen=True)
class BandConfig:
    """Weights, group membership, and threshold cut points.

    `calibrated` is honest bookkeeping: the initial thresholds are design
    guesses, and the eval harness refuses to report band metrics until
    calibration flips this to true.
    """

    version: int
    calibrated: bool
    weights: dict[str, int]
    groups: dict[TagGroup, frozenset[str]]
    thresholds: tuple[float, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BandConfig:
        """Build a config from parsed JSON, validating it against the taxonomy."""
        weights = {str(tag): int(weight) for tag, weight in payload["weights"].items()}
        missing = TAGS - weights.keys()
        if missing:
            raise ValueError(f"band config is missing weights for {sorted(missing)}")
        unknown = weights.keys() - TAGS
        if unknown:
            raise ValueError(f"band config weights contain unknown tags {sorted(unknown)}")
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("band config weights must be positive")

        groups: dict[TagGroup, frozenset[str]] = {}
        for name, members in payload["groups"].items():
            group = TagGroup(name)
            unknown_members = {str(m) for m in members} - TAGS
            if unknown_members:
                raise ValueError(f"group {name!r} contains unknown tags {sorted(unknown_members)}")
            groups[group] = frozenset(str(m) for m in members)

        for group in TagGroup:
            if group not in groups:
                raise ValueError(f"band config is missing group {group.value!r}")
        assigned = frozenset().union(*groups.values())
        if assigned != TAGS:
            raise ValueError("tags belong to no group or an unknown group")
        if sum(len(members) for members in groups.values()) != len(TAGS):
            raise ValueError("groups must form a disjoint partition of all tags")

        thresholds = tuple(float(t) for t in payload["thresholds"])
        expected = MAX_BAND - MIN_BAND
        if len(thresholds) != expected:
            raise ValueError(f"expected {expected} thresholds, got {len(thresholds)}")
        if list(thresholds) != sorted(thresholds):
            raise ValueError("thresholds must be ascending")

        return cls(
            version=_positive_version(payload["version"]),
            calibrated=bool(payload["calibrated"]),
            weights=weights,
            groups=groups,
            thresholds=thresholds,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> BandConfig:
        """Load a config from a JSON file."""
        with Path(path).open(encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
        return cls.from_dict(payload)

    def group_of(self, tag: str) -> TagGroup:
        """Group a tag belongs to, per this config rather than the code default."""
        for group, members in self.groups.items():
            if tag in members:
                return group
        raise KeyError(tag)

    def weight_of(self, tag: str) -> int:
        return self.weights[tag]

    def band_of(self, penalty: float) -> int:
        """Map a penalty onto a band. Higher penalty, lower band."""
        band = MAX_BAND
        for threshold in self.thresholds:
            if penalty > threshold:
                band -= 1
        return max(MIN_BAND, band)


def count_words(text: str) -> int:
    """Word count used to normalise penalties. Always the *stripped* text."""
    return len(text.split())


def penalty(
    edits: Iterable[Edit],
    group: TagGroup,
    word_count: int,
    config: BandConfig,
) -> float:
    """Summed weight of a group's edits, normalised by sqrt of the word count.

    Two errors in a six-word sentence is worse than two in thirty, hence the
    sqrt rather than a plain division.
    """
    total = sum(config.weight_of(edit.tag) for edit in edits if config.group_of(edit.tag) == group)
    if total == 0:
        return 0.0
    # An empty sentence has no words to spread the penalty across; treat it as
    # one so the penalty stays finite instead of dividing by zero.
    return total / math.sqrt(max(word_count, 1))


def derive_bands(
    edits: Sequence[Edit] | None,
    text: str,
    config: BandConfig,
) -> Bands:
    """Derive both bands from the edits over `text` (the stripped learner text).

    `edits is None` means the grader judged the sentence unparseable. That is
    the inverted-failure guard: an unreadable sentence yields no parseable edits,
    so the formula would score it 4 — the exact opposite of the truth. Grammar
    goes to 0 and the formula is skipped.
    """
    if edits is None:
        return Bands(grammar=MIN_BAND, naturalness=MIN_BAND)

    word_count = count_words(text)
    return Bands(
        grammar=config.band_of(penalty(edits, TagGroup.CORRECTNESS, word_count, config)),
        naturalness=config.band_of(penalty(edits, TagGroup.USAGE, word_count, config)),
    )


def default_config_path() -> Path:
    """Path to the `band_config.json` that ships in the repo root."""
    return Path(__file__).resolve().parents[2] / "band_config.json"


__all__ = [
    "MAX_BAND",
    "MIN_BAND",
    "BandConfig",
    "Bands",
    "count_words",
    "default_config_path",
    "derive_bands",
    "penalty",
]
