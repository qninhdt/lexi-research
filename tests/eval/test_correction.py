"""Correction metrics, against values computed by hand rather than by the code.

A metric that is subtly wrong still produces plausible numbers, and every result
downstream inherits the error silently. So the expected values here were worked
out on paper from the fixture below; a test that only asserted "greater than
zero" would pass against an implementation that scored spans backwards.

The fixture also carries the case the parent design's weight-tier property is
about: a right span with a confusable tag. That must not be counted the same as a
right span with a tag from another weight tier, because only the second can move
a band.
"""

from __future__ import annotations

import pytest

from lexi_research.eval.correction import (
    CorrectionScores,
    correction_scores,
    edit_triples,
    other_rate,
    tag_distribution,
)

# "She speak very well." — one edit at (4, 9), tag `tense`.
GOLD_ONE = "She [speak>spoke:tense] very well."
# Same span, same tag: a hit.
EXACT_ONE = "She [speak>spoke:tense] very well."
# Same span, tag from the same weight tier (`word` and `coll` are a confusable
# pair the taxonomy requires to carry equal weight).
SAME_TIER = "She [speak>spoke:word] very well."
# Same span, a tag from a different tier.
OTHER_TIER = "She [speak>spoke:sp] very well."
# A different span entirely.
WRONG_SPAN = "She speak [very>quite:word] well."
CLEAN = "She speak very well."


def test_edit_triples_reads_span_and_tag() -> None:
    assert edit_triples(GOLD_ONE) == [(4, 9, "tense")]


def test_an_unparseable_correction_contributes_no_edits() -> None:
    """Two graders who both declined agree; a crash here would say otherwise."""
    assert edit_triples("She [broken markup very well.") == []
    assert edit_triples(None) == []


def test_exact_f1_values(config) -> None:
    """One row, one gold edit, one predicted edit, matching: everything is 1.0."""
    scores = correction_scores([EXACT_ONE], [GOLD_ONE]).as_dict(config)
    assert scores["span_tag_precision"] == 1.0
    assert scores["span_tag_recall"] == 1.0
    assert scores["span_tag_f1"] == 1.0
    assert scores["span_only_f1"] == 1.0
    assert scores["tag_error_rate"] == 0.0


def test_span_only_vs_span_tag(config) -> None:
    """Right place, wrong name: span-only is perfect, span+tag is zero."""
    scores = correction_scores([SAME_TIER], [GOLD_ONE]).as_dict(config)
    assert scores["span_only_f1"] == 1.0
    assert scores["span_tag_f1"] == 0.0
    assert scores["tag_error_rate"] == 1.0


def test_a_wrong_span_scores_zero_on_both(config) -> None:
    scores = correction_scores([WRONG_SPAN], [GOLD_ONE]).as_dict(config)
    assert scores["span_only_f1"] == 0.0
    assert scores["span_tag_f1"] == 0.0


def test_two_clean_sentences_agree(config) -> None:
    """The easiest row in the dataset must not score zero."""
    scores = correction_scores([CLEAN], [CLEAN]).as_dict(config)
    assert scores["span_tag_f1"] == 1.0
    assert scores["exact_row_rate"] == 1.0


def test_a_missed_edit_and_a_spurious_one(config) -> None:
    """Hand-computed: 1 match, 2 predicted, 2 gold -> P=R=0.5, F1=0.5."""
    gold = ["She [speak>spoke:tense] very well.", "He [run>ran:tense] fast."]
    predicted = ["She [speak>spoke:tense] very well.", "He run [fast>quickly:word]."]
    scores = correction_scores(predicted, gold).as_dict(config)
    assert scores["span_tag_precision"] == 0.5
    assert scores["span_tag_recall"] == 0.5
    assert scores["span_tag_f1"] == 0.5


def test_precision_is_micro_averaged(config) -> None:
    """A hundred clean rows must not drown out the rows carrying edits.

    Macro-averaging would score this 0.5 (one perfect row, one zero row);
    micro-averaging counts edits, and there is exactly one gold edit missed.
    """
    gold = [CLEAN, GOLD_ONE]
    predicted = [CLEAN, WRONG_SPAN]
    scores = correction_scores(predicted, gold).as_dict(config)
    assert scores["gold_edits"] == 1
    assert scores["predicted_edits"] == 1
    assert scores["span_tag_f1"] == 0.0


def test_cross_tier_confusion(config) -> None:
    """`tense`→`word` share weight 2 and cannot move a band; `tense`→`sp` cannot not."""
    assert config.weight_of("tense") == config.weight_of("word")
    assert config.weight_of("tense") != config.weight_of("sp")

    assert correction_scores([SAME_TIER], [GOLD_ONE]).cross_tier_confusions(config) == 0
    assert correction_scores([OTHER_TIER], [GOLD_ONE]).cross_tier_confusions(config) == 1


def test_cross_tier_rate_is_over_matched_spans_only(config) -> None:
    """A wrong span is a span error, not a tag error, and must not dilute this."""
    scores = correction_scores([OTHER_TIER, WRONG_SPAN], [GOLD_ONE, GOLD_ONE]).as_dict(config)
    assert scores["cross_tier_confusions"] == 1
    assert scores["cross_tier_confusion_rate"] == 1.0


def test_confusable_pairs_are_never_cross_tier(config) -> None:
    """The taxonomy's own promise, measured through this metric."""
    from lexi_research.format.tags import CONFUSABLE_PAIRS

    for first, second in CONFUSABLE_PAIRS:
        gold = f"She [speak>spoke:{first}] very well."
        predicted = f"She [speak>spoke:{second}] very well."
        assert correction_scores([predicted], [gold]).cross_tier_confusions(config) == 0


def test_confusion_matrix_is_keyed_gold_to_predicted(config) -> None:
    scores = correction_scores([SAME_TIER], [GOLD_ONE]).as_dict(config)
    assert scores["confusion"] == {"tense->word": 1}


def test_tag_distribution_and_other_rate() -> None:
    corrections = [
        "She [speak>spoke:tense] very [well>good:other] today.",
        "He [run>ran:tense] fast.",
    ]
    counts = tag_distribution(corrections)
    assert counts == {"other": 1, "tense": 2}
    assert other_rate(counts) == pytest.approx(1 / 3)


def test_other_rate_of_nothing_is_zero() -> None:
    assert other_rate({}) == 0.0


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        correction_scores([CLEAN], [CLEAN, CLEAN])


def test_scores_accumulate_across_calls(config) -> None:
    scores = CorrectionScores()
    scores.add(edit_triples(EXACT_ONE), edit_triples(GOLD_ONE))
    scores.add(edit_triples(WRONG_SPAN), edit_triples(GOLD_ONE))
    payload = scores.as_dict(config)
    assert payload["gold_edits"] == 2
    assert payload["predicted_edits"] == 2
    # One of two matched: P = R = 0.5, so F1 = 0.5.
    assert payload["span_tag_f1"] == pytest.approx(0.5)


def test_evaluate_span_predictions_core_metrics() -> None:
    from lexi_research.eval.correction import compute_f_beta, evaluate_span_predictions

    # F0.5 formula check: P=1.0, R=0.5 -> F0.5 = 1.25 * (1 * 0.5) / (0.25 * 1 + 0.5) = 0.625 / 0.75 = 0.8333
    p, r, f05 = compute_f_beta(1, 1, 2, beta=0.5)
    assert p == 1.0
    assert r == 0.5
    assert f05 == pytest.approx(5 / 6)

    raw_inputs = [
        "He speak English.",
        "She speak very well.",
        "This is clean.",
    ]
    predictions = [
        "2 3 agr speaks",
        "2 3 word spoke",  # right span [2,3), wrong tag
        "OK",
    ]
    references = [
        "He [speak>speaks:agr] English.",
        "She [speak>spoke:tense] very well.",
        "This is clean.",
    ]

    metrics = evaluate_span_predictions(raw_inputs, predictions, references)
    # Full edit: 1 match out of 2 pred and 2 gold -> P=0.5, R=0.5 -> F0.5=0.5
    assert metrics["correction.full_edit_f05"] == pytest.approx(0.5)
    # Span detection: 2 matches out of 2 pred and 2 gold -> P=1.0, R=1.0 -> F0.5=1.0
    assert metrics["correction.span_f05"] == pytest.approx(1.0)
    # Clean sentence: 1 clean sentence predicted OK -> Clean Acc = 1.0
    assert metrics["correction.clean_accuracy"] == 1.0
    # Valid rate: all 3 outputs valid -> 1.0
    assert metrics["correction.valid_output_rate"] == 1.0

