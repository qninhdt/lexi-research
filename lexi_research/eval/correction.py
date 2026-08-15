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





def compute_f_beta(matched: int, predicted: int, gold: int, beta: float = 0.5) -> tuple[float, float, float]:
    """Compute Precision, Recall, and F-beta score (default beta=0.5). Two empty sets agree perfectly."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return precision, recall, 0.0
    f_beta = (1 + beta_sq) * (precision * recall) / denom
    return precision, recall, f_beta


def evaluate_span_predictions(
    raw_inputs: Sequence[str],
    predictions: Sequence[str | None],
    references: Sequence[str | None],
) -> dict[str, float]:
    """Evaluate 4 core metrics on span edit predictions against gold references.

    Metrics:
    1. Full Edit F0.5: Exact match on (start, end, tag, replacement)
    2. Span F0.5: Match on (start, end) detection only
    3. Clean Accuracy: Accuracy of predicting OK on clean sentences
    4. Valid Output Rate: Share of syntactically valid model outputs
    """
    from lexi_research.format.span_converter import (
        markup_to_spans,
        parse_span_output,
        validate_span_edits,
    )
    from lexi_research.format.units import SpanEdit, lex_units

    total_samples = len(raw_inputs) or 1
    valid_count = 0
    clean_total = 0
    clean_correct = 0

    total_full_pred = 0
    total_full_gold = 0
    total_full_matched = 0

    total_span_pred = 0
    total_span_gold = 0
    total_span_matched = 0

    for raw, pred_str, gold_str in zip(raw_inputs, predictions, references):
        raw_text = str(raw or "").strip()
        p_text = str(pred_str or "").strip()
        g_text = str(gold_str or "").strip()

        units = lex_units(raw_text)
        num_units = len(units)

        # Parse gold
        if "[" in g_text and "]" in g_text:
            g_spans_str = markup_to_spans(raw_text, g_text)
        else:
            g_spans_str = g_text

        g_parsed = parse_span_output(g_spans_str)
        is_gold_clean = g_parsed == "OK" or (isinstance(g_parsed, list) and len(g_parsed) == 0)
        if is_gold_clean:
            clean_total += 1

        # Parse pred
        is_valid, _ = validate_span_edits(p_text, num_units)
        if is_valid:
            valid_count += 1

        p_parsed = parse_span_output(p_text)
        is_pred_clean = p_parsed == "OK" or (isinstance(p_parsed, list) and len(p_parsed) == 0)

        if is_gold_clean and is_pred_clean:
            clean_correct += 1

        # Convert to sets of tuples
        p_full_set: set[tuple[int, int, str, str]] = set()
        p_span_set: set[tuple[int, int]] = set()
        if isinstance(p_parsed, list):
            for e in p_parsed:
                p_full_set.add((e.start, e.end, e.tag, e.replacement))
                p_span_set.add((e.start, e.end))

        g_full_set: set[tuple[int, int, str, str]] = set()
        g_span_set: set[tuple[int, int]] = set()
        if isinstance(g_parsed, list):
            for e in g_parsed:
                g_full_set.add((e.start, e.end, e.tag, e.replacement))
                g_span_set.add((e.start, e.end))

        total_full_pred += len(p_full_set)
        total_full_gold += len(g_full_set)
        total_full_matched += len(p_full_set & g_full_set)

        total_span_pred += len(p_span_set)
        total_span_gold += len(g_span_set)
        total_span_matched += len(p_span_set & g_span_set)

    _, _, full_f05 = compute_f_beta(total_full_matched, total_full_pred, total_full_gold, beta=0.5)
    _, _, span_f05 = compute_f_beta(total_span_matched, total_span_pred, total_span_gold, beta=0.5)

    clean_acc = (clean_correct / clean_total) if clean_total > 0 else 1.0
    valid_rate = valid_count / total_samples

    return {
        "correction.full_edit_f05": full_f05,
        "correction.span_f05": span_f05,
        "correction.clean_accuracy": clean_acc,
        "correction.valid_output_rate": valid_rate,
    }


def evaluate_rewrite_predictions(
    raw_inputs: Sequence[str],
    predictions: Sequence[str | None],
    references: Sequence[str | None],
) -> dict[str, float]:
    """Evaluate Pass 1 rewritten predictions against gold references using the Canonical Aligner.

    Metrics:
    1. Correction Edit F0.5: Precision, Recall, and F0.5 over (start, end, replacement) edits.
    2. Clean Exact Accuracy: Accuracy of keeping clean sentences verbatim (prediction == raw).
    3. Sentence Exact Match: Share of predictions matching gold corrected sentence exactly.
    4. Null Accuracy: Accuracy of predicting 'null' when gold is 'null'.
    """
    from lexi_research.format.aligner import align_words, annotated_to_corrected

    total_samples = len(raw_inputs) or 1
    clean_total = 0
    clean_correct = 0
    null_total = 0
    null_correct = 0
    exact_match_count = 0

    total_pred_edits = 0
    total_gold_edits = 0
    total_matched_edits = 0

    for raw, pred_str, gold_str in zip(raw_inputs, predictions, references):
        raw_text = str(raw or "").strip()
        p_text = str(pred_str or "").strip()
        g_ref = str(gold_str or "").strip()

        # Derive gold corrected sentence
        g_corrected = annotated_to_corrected(g_ref) if ("[" in g_ref and "]" in g_ref) else g_ref

        # Check null accuracy
        is_gold_null = g_corrected.lower() == "null" or g_ref.lower() == "null"
        is_pred_null = p_text.lower() == "null"

        if is_gold_null:
            null_total += 1
            if is_pred_null:
                null_correct += 1
            continue

        if is_pred_null:
            gold_edits = align_words(raw_text, g_corrected)
            total_gold_edits += len(gold_edits)
            continue

        # Check sentence exact match
        if p_text == g_corrected:
            exact_match_count += 1

        # Check clean sentence accuracy
        is_clean_gold = g_corrected == raw_text
        if is_clean_gold:
            clean_total += 1
            if p_text == raw_text:
                clean_correct += 1

        # Extract edits via the EXACT SAME canonical aligner
        pred_edits = align_words(raw_text, p_text)
        gold_edits = align_words(raw_text, g_corrected)

        p_set: set[tuple[int, int, str]] = {
            (e.start, e.end, e.replacement.strip()) for e in pred_edits
        }
        g_set: set[tuple[int, int, str]] = {
            (e.start, e.end, e.replacement.strip()) for e in gold_edits
        }

        total_pred_edits += len(p_set)
        total_gold_edits += len(g_set)
        total_matched_edits += len(p_set & g_set)

    p, r, f05 = compute_f_beta(total_matched_edits, total_pred_edits, total_gold_edits, beta=0.5)
    clean_acc = (clean_correct / clean_total) if clean_total > 0 else 1.0
    exact_match_rate = exact_match_count / total_samples
    null_acc = (null_correct / null_total) if null_total > 0 else 1.0

    return {
        "correction.edit_f05": f05,
        "correction.edit_precision": p,
        "correction.edit_recall": r,
        "correction.clean_accuracy": clean_acc,
        "correction.exact_match": exact_match_rate,
        "correction.null_accuracy": null_acc,
    }


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
    "compute_f_beta",
    "correction_scores",
    "edit_triples",
    "evaluate_rewrite_predictions",
    "evaluate_span_predictions",
    "other_rate",
    "tag_distribution",
]
