"""Correction metrics that separate *wrong place* from *wrong name*.

A single edit-F1 collapses two very different failures. A student that finds the
right span and calls it `coll` instead of `word` has understood the sentence; one
that marks the wrong three words has not. Only the second is a real error, and
the parent design's weight-tier property is what makes the distinction principled:
tags carrying equal weight cannot move a band, so confusing them is harmless by
construction.

So three numbers instead of one — span+tag F1, span-only F1, and the share of
matched spans whose tags fall in *different* weight tiers. The gap between the
first two is tag error; the third says how much of that tag error could have
changed a grade.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lexi_research.format import BandConfig
from lexi_research.format.parser import ParseError, parse_correction

#: `(start, end, tag)` against the stripped learner text.
Triple = tuple[int, int, str]


def edit_triples(correction: str | None) -> list[Triple]:
    """Edits from a correction string; an unparseable one contributes none.

    That is the right reading for agreement rather than a silent failure: two
    graders who both declined to correct a sentence agree, and both produce the
    empty set.
    """
    if correction is None:
        return []
    parsed = parse_correction(correction)
    if isinstance(parsed, ParseError):
        return []
    return [(edit.span[0], edit.span[1], edit.tag) for edit in parsed.edits]


def _prf(matched: int, predicted: int, gold: int) -> tuple[float, float, float]:
    """Precision, recall, F1. Two empty sets agree perfectly."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


@dataclass
class CorrectionScores:
    """Accumulated over a dataset, so precision is micro-averaged, not per-row.

    Micro rather than macro: a per-row average lets a hundred clean sentences,
    each scoring a free 1.0, drown out the rows that carry edits.
    """

    span_tag_matched: int = 0
    span_matched: int = 0
    predicted: int = 0
    gold: int = 0
    rows: int = 0
    exact_rows: int = 0
    #: `(gold_tag, predicted_tag) -> count`, over spans both sides agreed on.
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, predicted: Sequence[Triple], gold: Sequence[Triple]) -> None:
        predicted_set, gold_set = set(predicted), set(gold)
        self.rows += 1
        self.predicted += len(predicted_set)
        self.gold += len(gold_set)
        self.span_tag_matched += len(predicted_set & gold_set)
        if predicted_set == gold_set:
            self.exact_rows += 1

        gold_spans: dict[tuple[int, int], str] = {(s, e): tag for s, e, tag in gold_set}
        for start, end, tag in sorted(predicted_set):
            gold_tag = gold_spans.pop((start, end), None)
            if gold_tag is None:
                continue
            self.span_matched += 1
            key = (gold_tag, tag)
            self.confusion[key] = self.confusion.get(key, 0) + 1

    def cross_tier_confusions(self, config: BandConfig) -> int:
        """Matched spans whose tags carry different weights.

        Equal weight means the confusion cannot move a band — the property the
        taxonomy asserts for its confusable pairs — so those are excluded here
        rather than counted and then explained away in prose.
        """
        return sum(
            count
            for (gold_tag, predicted_tag), count in self.confusion.items()
            if config.weight_of(gold_tag) != config.weight_of(predicted_tag)
        )

    def as_dict(self, config: BandConfig) -> dict[str, Any]:
        span_tag_p, span_tag_r, span_tag_f1 = _prf(self.span_tag_matched, self.predicted, self.gold)
        span_p, span_r, span_f1 = _prf(self.span_matched, self.predicted, self.gold)
        cross = self.cross_tier_confusions(config)
        return {
            "span_tag_precision": span_tag_p,
            "span_tag_recall": span_tag_r,
            "span_tag_f1": span_tag_f1,
            "span_only_precision": span_p,
            "span_only_recall": span_r,
            "span_only_f1": span_f1,
            # The share of right-place edits whose tag was wrong. The gap between
            # the two F1s expressed as the number a reader actually wants.
            "tag_error_rate": (
                (self.span_matched - self.span_tag_matched) / self.span_matched
                if self.span_matched
                else 0.0
            ),
            "cross_tier_confusion_rate": cross / self.span_matched if self.span_matched else 0.0,
            "cross_tier_confusions": cross,
            "exact_row_rate": self.exact_rows / self.rows if self.rows else 0.0,
            "predicted_edits": self.predicted,
            "gold_edits": self.gold,
            "confusion": {
                f"{gold}->{pred}": count for (gold, pred), count in self.confusion.items()
            },
        }


def correction_scores(
    predicted: Iterable[str | None],
    gold: Iterable[str | None],
) -> CorrectionScores:
    """Score a dataset of correction strings against the teacher's."""
    scores = CorrectionScores()
    for prediction, reference in zip(predicted, gold, strict=True):
        scores.add(edit_triples(prediction), edit_triples(reference))
    return scores


def tag_distribution(corrections: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for correction in corrections:
        for _, _, tag in edit_triples(correction):
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def other_rate(counts: Mapping[str, int]) -> float:
    """Share of edits tagged `other` — how much the taxonomy is failing to cover.

    Compared against the teacher's own rate rather than against zero: a student
    that reproduces the teacher's taxonomy gaps is distilling faithfully, and
    one that reports fewer `other`s may simply be guessing a specific tag.
    """
    total = sum(counts.values())
    return counts.get("other", 0) / total if total else 0.0


__all__ = [
    "CorrectionScores",
    "Triple",
    "correction_scores",
    "edit_triples",
    "other_rate",
    "tag_distribution",
]
