"""Band-calculator tests: thresholds, length normalisation, group isolation.

The band formula is the part of the design that stays tunable after training, so
these tests pin its *shape* — monotonicity, isolation between the two groups, the
inverted-failure guard — rather than the placeholder cut points, which Phase 6
replaces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from lexi_research.format import (
    MAX_BAND,
    MIN_BAND,
    BandConfig,
    Edit,
    ParseOk,
    Tag,
    TagGroup,
    count_words,
    default_config_path,
    derive_bands,
    parse_correction,
    penalty,
)


def edit(tag: Tag | str, start: int = 0, end: int = 0) -> Edit:
    """An edit carrying only what the band formula reads: its tag."""
    return Edit(original="", replacement="x", tag=str(tag), span=(start, end))


def raw_config() -> dict[str, Any]:
    with default_config_path().open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def test_ships_uncalibrated_and_says_so(config: BandConfig) -> None:
    """The placeholder thresholds must be flagged, so eval can refuse to report."""
    assert config.calibrated is False
    assert config.version == 1


def test_clean_sentence_scores_top_band(config: BandConfig) -> None:
    bands = derive_bands([], "The room has bright light in the morning.", config)
    assert bands.grammar == MAX_BAND
    assert bands.naturalness == MAX_BAND


def test_null_correction_drops_grammar_to_zero(config: BandConfig) -> None:
    """The inverted-failure guard.

    An unreadable sentence yields no parseable edits, so the formula alone would
    score it top band — the exact opposite of the truth. `None` is the escape
    hatch the grader uses instead.
    """
    bands = derive_bands(None, "word word word", config)
    assert bands.grammar == MIN_BAND
    assert bands.naturalness == MIN_BAND


def test_usage_errors_leave_grammar_untouched(config: BandConfig) -> None:
    text = "I want to do a decision about this today"
    bands = derive_bands([edit(Tag.COLL), edit(Tag.UNNAT)], text, config)
    assert bands.grammar == MAX_BAND
    assert bands.naturalness < MAX_BAND


def test_correctness_errors_leave_naturalness_untouched(config: BandConfig) -> None:
    text = "He speak very eloquent in the morning time"
    bands = derive_bands([edit(Tag.AGR), edit(Tag.FORM), edit(Tag.ORDER)], text, config)
    assert bands.naturalness == MAX_BAND
    assert bands.grammar < MAX_BAND


def test_other_counts_against_grammar_not_naturalness(config: BandConfig) -> None:
    """A catch-all error is more likely grammatical than stylistic."""
    assert config.group_of(Tag.OTHER.value) is TagGroup.CORRECTNESS


def test_more_errors_never_raise_the_band(config: BandConfig) -> None:
    text = " ".join(["word"] * 12)
    previous = MAX_BAND + 1
    for count in range(0, 7):
        band = derive_bands([edit(Tag.AGR) for _ in range(count)], text, config).grammar
        assert band <= previous
        previous = band
    assert previous == MIN_BAND, "enough errors must reach the bottom band"


def test_shorter_sentence_is_punished_harder(config: BandConfig) -> None:
    """Two errors in six words is worse than two in thirty."""
    edits = [edit(Tag.AGR), edit(Tag.ART)]
    short = penalty(edits, TagGroup.CORRECTNESS, 6, config)
    long = penalty(edits, TagGroup.CORRECTNESS, 30, config)
    assert short > long
    assert (
        derive_bands(edits, " ".join(["w"] * 6), config).grammar
        <= derive_bands(edits, " ".join(["w"] * 30), config).grammar
    )


def test_penalty_matches_the_documented_formula(config: BandConfig) -> None:
    edits = [edit(Tag.AGR), edit(Tag.ART), edit(Tag.ORDER)]  # 2 + 1 + 3
    assert penalty(edits, TagGroup.CORRECTNESS, 9, config) == pytest.approx(6 / math.sqrt(9))


def test_penalty_of_an_empty_edit_list_is_zero(config: BandConfig) -> None:
    assert penalty([], TagGroup.CORRECTNESS, 10, config) == 0.0
    assert penalty([], TagGroup.USAGE, 10, config) == 0.0


def test_empty_text_does_not_divide_by_zero(config: BandConfig) -> None:
    assert penalty([edit(Tag.AGR)], TagGroup.CORRECTNESS, 0, config) == pytest.approx(2.0)
    assert derive_bands([edit(Tag.AGR)], "", config).grammar <= MAX_BAND


@pytest.mark.parametrize("index", range(4))
def test_band_flips_exactly_at_each_threshold(config: BandConfig, index: int) -> None:
    """A penalty at the cut point keeps the higher band; just past it drops one."""
    threshold = config.thresholds[index]
    epsilon = 1e-9
    assert config.band_of(threshold) == MAX_BAND - index
    assert config.band_of(threshold + epsilon) == MAX_BAND - index - 1


def test_band_never_leaves_the_zero_to_four_range(config: BandConfig) -> None:
    for value in (-5.0, 0.0, 0.001, 0.5, 1.0, 2.0, 100.0):
        assert MIN_BAND <= config.band_of(value) <= MAX_BAND


def test_word_count_uses_the_stripped_text(config: BandConfig) -> None:
    """Counting the marked-up string would inflate the word count and inflate the band."""
    correction = "The room [have>has:agr] bright light in [>the:art] morning."
    parsed = parse_correction(correction)
    assert isinstance(parsed, ParseOk)

    assert count_words(parsed.text) == 7
    assert count_words(correction) > count_words(parsed.text)

    from_stripped = derive_bands(parsed.edits, parsed.text, config)
    from_raw = derive_bands(parsed.edits, correction, config)
    assert from_stripped.grammar <= from_raw.grammar


def test_count_words_ignores_extra_whitespace() -> None:
    assert count_words("  two   words \n") == 2
    assert count_words("") == 0


def test_config_rejects_a_missing_weight() -> None:
    payload = raw_config()
    del payload["weights"]["agr"]
    with pytest.raises(ValueError, match="missing weights"):
        BandConfig.from_dict(payload)


def test_config_rejects_a_tag_outside_the_taxonomy() -> None:
    payload = raw_config()
    payload["weights"]["invented"] = 2
    with pytest.raises(ValueError, match="unknown tags"):
        BandConfig.from_dict(payload)


def test_config_rejects_a_tag_in_no_group() -> None:
    payload = raw_config()
    payload["groups"]["correctness"].remove("punc")
    with pytest.raises(ValueError, match="no group"):
        BandConfig.from_dict(payload)


def test_config_rejects_descending_thresholds() -> None:
    payload = raw_config()
    payload["thresholds"] = [0.9, 0.4, 0.0, 1.6]
    with pytest.raises(ValueError, match="ascending"):
        BandConfig.from_dict(payload)


def test_config_rejects_the_wrong_number_of_thresholds() -> None:
    payload = raw_config()
    payload["thresholds"] = [0.0, 0.5]
    with pytest.raises(ValueError, match="thresholds"):
        BandConfig.from_dict(payload)


def test_default_config_path_points_at_the_shipped_file() -> None:
    path: Path = default_config_path()
    assert path.name == "band_config.json"
    assert path.is_file()
