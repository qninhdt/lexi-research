"""Stage-A import invariants.

The conversion is mechanical, which is exactly why it needs tests: a
detokenisation bug produces a dataset that trains without error and teaches the
model to emit spacing no learner writes. Every test here pins a decision that was
made against measured corpus behaviour rather than a guess.
"""

from __future__ import annotations

import pytest

from lexi_research.data.gec_import import (
    ERRANT_TO_TAG,
    GecImportError,
    ImportStats,
    M2Sentence,
    cap_strata,
    convert_sentence,
    dedupe,
    detokenise,
    detokenise_text,
    drop_fragments,
    edit_bucket,
    read_m2,
    row_uid,
    split_rows,
    tag_for,
)
from lexi_research.format import TAGS, parse_correction
from lexi_research.format.parser import ParseError


def sentence(tokens: str, *annotations: tuple[int, int, str, str]) -> M2Sentence:
    """An M2 sentence from a token string plus `(start, end, type, correction)`."""
    return M2Sentence(
        tokens=tuple(tokens.split(" ")),
        annotations=tuple((a, b, t, c, "0") for a, b, t, c in annotations),
    )


class TestDetokenise:
    def test_punctuation_attaches_to_the_preceding_word(self) -> None:
        text, _ = detokenise(["stores", ".", "There", "are", "banks", ",", "bars", "."])
        assert text == "stores. There are banks, bars."

    def test_clitics_attach_without_a_space(self) -> None:
        text, _ = detokenise(["I", "do", "n't", "like", "John", "'s", "book"])
        assert text == "I don't like John's book"

    def test_offsets_index_the_returned_text(self) -> None:
        tokens = ["The", "room", "is", "bright", "."]
        text, offsets = detokenise(tokens)
        for index, token in enumerate(tokens):
            assert text[offsets[index] : offsets[index] + len(token)] == token
        assert offsets[len(tokens)] == len(text)

    def test_offsets_disambiguate_a_repeated_word(self) -> None:
        """Searching for the token would find the first occurrence, not the right one."""
        tokens = ["the", "cat", "and", "the", "dog"]
        text, offsets = detokenise(tokens)
        assert text[offsets[3] : offsets[3] + 3] == "the"
        assert offsets[3] != offsets[0]

    def test_replacement_strings_are_detokenised_too(self) -> None:
        """2.26% of M2 corrections carry token spacing; untouched they enter the markup."""
        assert detokenise_text(". However ,") == ". However,"

    def test_empty_replacement_stays_empty(self) -> None:
        assert detokenise_text("") == ""


class TestTagMapping:
    def test_every_mapped_tag_is_in_the_taxonomy(self) -> None:
        assert set(ERRANT_TO_TAG.values()) <= TAGS

    @pytest.mark.parametrize("prefix", ["M", "U", "R"])
    def test_the_operation_prefix_is_discarded(self, prefix: str) -> None:
        """Which operation an edit performs is carried by which side is empty."""
        assert tag_for(f"{prefix}:DET", "the", "") == "art"

    def test_multi_token_other_becomes_unnat(self) -> None:
        """71% of R:OTHER is phrasing, which is what `unnat` means."""
        assert tag_for("R:OTHER", "more memories bring me", "brought me the most") == "unnat"

    def test_single_token_other_becomes_word(self) -> None:
        assert tag_for("R:OTHER", "pursue", "seek") == "word"

    def test_an_unmapped_type_returns_none(self) -> None:
        assert tag_for("R:NONSENSE", "a", "b") is None

    def test_collocation_is_never_produced(self) -> None:
        """ERRANT has no collocation type; `coll` is taught in stage B instead."""
        assert "coll" not in ERRANT_TO_TAG.values()


class TestConvertSentence:
    def test_round_trips_to_the_original_text(self) -> None:
        result = convert_sentence(sentence("He speak well .", (1, 2, "R:VERB:SVA", "speaks")))
        assert result is not None
        text, correction, tags = result
        assert text == "He speak well."
        assert tags == ["agr"]
        parsed = parse_correction(correction)
        assert not isinstance(parsed, ParseError)
        assert parsed.text == text

    def test_a_clean_sentence_is_re_emitted_verbatim(self) -> None:
        result = convert_sentence(sentence("This is fine ."))
        assert result == ("This is fine.", "This is fine.", [])

    def test_unk_drops_the_edit_but_keeps_the_sentence(self) -> None:
        """UNK annotations in this corpus are no-ops: 'chess' -> 'chess'."""
        stats = ImportStats()
        result = convert_sentence(
            sentence("I play chess often .", (2, 3, "UNK", "chess")), stats=stats
        )
        assert result is not None
        _, correction, tags = result
        assert tags == []
        assert correction == "I play chess often."
        assert stats.skipped_edits["UNK"] == 1

    def test_noop_is_skipped_the_same_way(self) -> None:
        result = convert_sentence(sentence("All good .", (0, 0, "noop", "-NONE-")))
        assert result == ("All good.", "All good.", [])

    def test_an_unmapped_type_drops_the_whole_sentence(self) -> None:
        """Dropping just the edit would present a wrong sentence as already correct."""
        stats = ImportStats()
        result = convert_sentence(sentence("A b c .", (1, 2, "R:NONSENSE", "x")), stats=stats)
        assert result is None
        assert stats.dropped["unmapped_type:R:NONSENSE"] == 1

    def test_overlapping_edits_are_rejected(self) -> None:
        stats = ImportStats()
        result = convert_sentence(
            sentence(
                "I go to the shop .",
                (1, 3, "R:VERB", "went to"),
                (2, 4, "R:PREP", "into the"),
            ),
            stats=stats,
        )
        assert result is None
        assert stats.dropped["overlapping_edits"] == 1

    def test_a_deletion_renders_with_an_empty_replacement(self) -> None:
        result = convert_sentence(sentence("I like the music .", (2, 3, "U:DET", "")))
        assert result is not None
        _, correction, _ = result
        assert "[the>:art]" in correction

    def test_an_insertion_renders_with_an_empty_original(self) -> None:
        result = convert_sentence(sentence("I went school .", (2, 2, "M:PREP", "to")))
        assert result is not None
        text, correction, _ = result
        assert "[>to:prep]" in correction
        parsed = parse_correction(correction)
        assert not isinstance(parsed, ParseError)
        assert parsed.text == text

    def test_a_replacement_carrying_token_spacing_is_detokenised(self) -> None:
        result = convert_sentence(sentence("Good , however I went .", (1, 2, "R:PUNCT", ". And ,")))
        assert result is not None
        _, correction, _ = result
        assert ". And," in correction

    def test_a_span_beyond_the_sentence_is_rejected(self) -> None:
        stats = ImportStats()
        result = convert_sentence(sentence("Short .", (5, 9, "R:VERB", "x")), stats=stats)
        assert result is None
        assert stats.dropped["span_out_of_range"] == 1

    def test_annotations_from_another_coder_are_ignored(self) -> None:
        """Merging annotators produces overlaps that cannot render."""
        merged = M2Sentence(
            tokens=("He", "speak", "."),
            annotations=(
                (1, 2, "R:VERB:SVA", "speaks", "0"),
                (1, 2, "R:VERB:TENSE", "spoke", "1"),
            ),
        )
        result = convert_sentence(merged, coder="0")
        assert result is not None
        assert result[2] == ["agr"]


class TestStrata:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [(0, "clean"), (1, "one"), (2, "few"), (3, "few"), (4, "many"), (40, "many")],
    )
    def test_bucket_boundaries(self, count: int, expected: str) -> None:
        assert edit_bucket(count) == expected

    def _rows(self, clean: int, other: int) -> list[dict[str, object]]:
        rows = [
            {"row_uid": f"c{i:04d}", "edit_bucket": "clean", "cefr": "A"} for i in range(clean)
        ]
        rows += [{"row_uid": f"o{i:04d}", "edit_bucket": "one", "cefr": "B"} for i in range(other)]
        return rows

    def test_a_dominant_stratum_is_capped(self) -> None:
        capped = cap_strata(self._rows(900, 100), max_stratum_share=0.15, seed=1)
        buckets = [row["edit_bucket"] for row in capped]
        assert buckets.count("clean") == 150

    def test_a_rare_stratum_is_never_discarded(self) -> None:
        capped = cap_strata(self._rows(900, 3), max_stratum_share=0.15, seed=1)
        assert [row["edit_bucket"] for row in capped].count("one") == 3

    def test_the_cap_is_deterministic(self) -> None:
        rows = self._rows(900, 100)
        first = cap_strata(rows, max_stratum_share=0.15, seed=7)
        second = cap_strata(rows, max_stratum_share=0.15, seed=7)
        assert [row["row_uid"] for row in first] == [row["row_uid"] for row in second]

    def test_a_different_seed_selects_differently(self) -> None:
        rows = self._rows(900, 100)
        first = cap_strata(rows, max_stratum_share=0.15, seed=1)
        second = cap_strata(rows, max_stratum_share=0.15, seed=2)
        assert [row["row_uid"] for row in first] != [row["row_uid"] for row in second]

    @pytest.mark.parametrize("share", [0.0, 1.5, -0.2])
    def test_an_out_of_range_share_raises(self, share: float) -> None:
        with pytest.raises(ValueError, match="max_stratum_share"):
            cap_strata(self._rows(10, 10), max_stratum_share=share, seed=1)


class TestDedupeAndSplit:
    def test_duplicate_rows_collapse_to_one(self) -> None:
        uid = row_uid("Thank you.", "Thank you.")
        rows = [{"row_uid": uid, "text": "Thank you."} for _ in range(4)]
        assert len(dedupe(rows)) == 1

    def test_row_uid_is_content_addressed(self) -> None:
        assert row_uid("a", "b") == row_uid("a", "b")
        assert row_uid("a", "b") != row_uid("a", "c")

    def test_split_is_deterministic_and_covers_every_row(self) -> None:
        rows = [{"row_uid": f"{i:04d}"} for i in range(500)]
        first = split_rows(rows, seed=3, val_share=0.1)
        second = split_rows(rows, seed=3, val_share=0.1)
        assert [row["split"] for row in first] == [row["split"] for row in second]
        assert {row["split"] for row in first} == {"train", "val"}

    def test_val_share_is_approximately_honoured(self) -> None:
        rows = [{"row_uid": f"{i:05d}"} for i in range(5000)]
        split = split_rows(rows, seed=3, val_share=0.1)
        share = sum(1 for row in split if row["split"] == "val") / len(split)
        assert 0.08 < share < 0.12

    @pytest.mark.parametrize("share", [0.0, 1.0, -0.1])
    def test_an_out_of_range_val_share_raises(self, share: float) -> None:
        with pytest.raises(ValueError, match="val_share"):
            split_rows([{"row_uid": "a"}], seed=1, val_share=share)


class TestReadM2:
    def test_reads_a_sentence_and_its_annotations(self, tmp_path) -> None:
        path = tmp_path / "x.m2"
        path.write_text(
            "S He speak .\nA 1 2|||R:VERB:SVA|||speaks|||REQUIRED|||-NONE-|||0\n\n"
            "S All good .\n",
            encoding="utf-8",
        )
        parsed = list(read_m2(path))
        assert len(parsed) == 2
        assert parsed[0].tokens == ("He", "speak", ".")
        assert parsed[0].annotations[0][2] == "R:VERB:SVA"
        assert parsed[1].annotations == ()

    def test_an_annotation_before_any_sentence_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.m2"
        path.write_text("A 0 1|||R:DET|||the|||REQUIRED|||-NONE-|||0\n", encoding="utf-8")
        with pytest.raises(GecImportError, match="precedes any S line"):
            list(read_m2(path))

    def test_a_malformed_annotation_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.m2"
        path.write_text("S He speak .\nA 1 2|||R:VERB:SVA\n", encoding="utf-8")
        with pytest.raises(GecImportError, match="malformed annotation"):
            list(read_m2(path))


class TestFragments:
    def test_a_bare_name_is_dropped(self) -> None:
        """W&I is essays split into sentences; the split leaves headings behind."""
        rows = [
            {"n_words": 1, "text": "Svetlana"},
            {"n_words": 2, "text": "Take care,"},
            {"n_words": 6, "text": "This one is a real sentence"},
        ]
        kept = drop_fragments(rows, min_words=3)
        assert [row["text"] for row in kept] == ["This one is a real sentence"]

    def test_min_words_of_one_keeps_everything(self) -> None:
        rows = [{"n_words": 1, "text": "Hi"}]
        assert drop_fragments(rows, min_words=1) == rows
