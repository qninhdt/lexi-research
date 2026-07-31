"""Tests for distinct-n.

The measurement exists to catch one specific failure: a batched call returning K
paraphrases of one sentence. The tests below pin the two ends of that scale and
the boundary conditions that would otherwise produce a misleading number.
"""

from __future__ import annotations

import pytest

from lexi_research.data.diversity import (
    batch_diversity,
    distinct_n,
    ngrams,
    tokenize,
)


class TestTokenize:
    def test_lowercases(self) -> None:
        assert tokenize("The Room") == ["the", "room"]

    def test_drops_punctuation(self) -> None:
        """Two sentences differing only in a comma are not diverse."""
        assert tokenize("bright, airy!") == ["bright", "airy"]

    def test_keeps_apostrophes(self) -> None:
        """`don't` and `do not` are different learner behaviour, not noise."""
        assert tokenize("don't") == ["don't"]

    def test_keeps_digits(self) -> None:
        assert tokenize("3 books") == ["3", "books"]

    def test_empty_text_yields_no_tokens(self) -> None:
        assert tokenize("   ") == []


class TestNgrams:
    def test_bigrams(self) -> None:
        assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]

    def test_text_shorter_than_n_yields_none(self) -> None:
        assert ngrams(["a"], 2) == []

    def test_n_equal_to_length_yields_one(self) -> None:
        assert ngrams(["a", "b"], 2) == [("a", "b")]

    def test_zero_n_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ngrams(["a"], 0)


class TestDistinctN:
    def test_identical_sentences_score_low(self) -> None:
        """The failure mode: six copies of one sentence."""
        texts = ["the room was bright and airy"] * 6
        assert distinct_n(texts, 2) == pytest.approx(1 / 6)

    def test_fully_distinct_sentences_score_one(self) -> None:
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        assert distinct_n(texts, 2) == 1.0

    def test_measures_repetition_between_sentences(self) -> None:
        """A per-sentence average would call six identical sentences diverse."""
        pooled = distinct_n(["a b c d", "a b c d"], 2)
        single = distinct_n(["a b c d"], 2)
        assert pooled < single

    def test_empty_pool_scores_one(self) -> None:
        """No evidence of collapse is not evidence of collapse."""
        assert distinct_n([], 2) == 1.0

    def test_sentences_shorter_than_n_score_one(self) -> None:
        assert distinct_n(["hi", "yo"], 3) == 1.0

    def test_trigrams_are_stricter_than_bigrams(self) -> None:
        texts = ["the cat sat on the mat", "the cat sat on a rug"]
        assert distinct_n(texts, 3) >= distinct_n(texts, 2)


class TestBatchDiversity:
    def test_flags_a_collapsed_batch(self) -> None:
        measure = batch_diversity("b1", ["same words here"] * 6, threshold=0.7)
        assert measure.is_collapsed

    def test_does_not_flag_a_varied_batch(self) -> None:
        texts = [
            "the lamp gives a bright glow",
            "she opened every window this morning",
            "sunlight filled my small kitchen",
            "we walked along a quiet river",
            "his notebook fell under the desk",
            "they cooked dinner without any salt",
        ]
        measure = batch_diversity("b2", texts, threshold=0.7)
        assert not measure.is_collapsed

    def test_carries_the_batch_id(self) -> None:
        assert batch_diversity("b3", ["a b c"]).batch_uid == "b3"

    def test_records_the_text_count(self) -> None:
        assert batch_diversity("b4", ["a b", "c d", "e f"]).texts == 3

    def test_empty_batch_is_not_flagged(self) -> None:
        """An empty batch failed for other reasons; do not double-report it."""
        assert not batch_diversity("b5", []).is_collapsed

    def test_reports_both_orders(self) -> None:
        measure = batch_diversity("b6", ["one two three four", "five six seven eight"])
        assert 0.0 <= measure.distinct2 <= 1.0
        assert 0.0 <= measure.distinct3 <= 1.0

    def test_as_dict_is_report_shaped(self) -> None:
        payload = batch_diversity("b7", ["a b c"]).as_dict()
        assert set(payload) == {"batch_uid", "texts", "distinct2", "distinct3", "collapsed"}
