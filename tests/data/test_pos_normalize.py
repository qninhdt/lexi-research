from lexi_research.data.pos_normalize import (
    has_placeholder,
    is_explicitly_excluded,
    is_multiword,
    normalize_pos,
    normalize_target,
)


def test_all_observed_pos_values_are_mapped_or_explicitly_excluded() -> None:
    observed = {
        "noun", "adjective", "verb", "idiom", "adverb", "phrasal verb", "collocation", "phrase",
        "exclamation", "plural noun", "", "suffix", "preposition", "prefix", "pronoun", "number",
        "conjunction", "determiner", "ordinal number", "modal verb", "predeterminer", "auxiliary verb",
        "abbreviation", "interjection", "adjective, adverb", "adj", "adverb, adjective", "written abbreviation",
        "trademark", "symbol", "short form", "noun or exclamation", "n", "modifier", "indefinite article",
        "definite article", "combining form", "V",
    }
    assert all(normalize_pos(value) is not None or is_explicitly_excluded(value) for value in observed)


def test_normalization_and_flags() -> None:
    assert normalize_pos("V") == "verb"
    assert normalize_pos("plural noun") == "noun"
    assert normalize_pos("suffix") is None
    assert normalize_target("  Credit   Card ") == "credit card"
    assert is_multiword("credit card", "noun")
    assert is_multiword("put up", "phrasal verb")
    assert has_placeholder("put sb down")
    assert not has_placeholder("substance")
