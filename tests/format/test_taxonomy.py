"""Taxonomy invariants.

These tests exist to stop a future taxonomy edit from silently breaking the
property that makes band-derivation viable: tags a grader routinely confuses
carry equal weight, so confusing them cannot move a band.
"""

from __future__ import annotations

import pytest

from lexi_research.format import (
    CONFUSABLE_PAIRS,
    GROUP_OF,
    TAGS,
    BandConfig,
    Tag,
    TagGroup,
    group_of,
)


def test_the_set_holds_exactly_sixteen_tags() -> None:
    assert len(TAGS) == 16
    assert len(list(Tag)) == 16


def test_every_tag_belongs_to_exactly_one_group() -> None:
    assert set(GROUP_OF) == set(Tag)
    for tag in Tag:
        assert GROUP_OF[tag] in TagGroup


@pytest.mark.parametrize(("left", "right"), CONFUSABLE_PAIRS)
def test_confusable_tags_share_a_weight(config: BandConfig, left: Tag, right: Tag) -> None:
    """The load-bearing invariant: confusing these pairs cannot shift a band."""
    assert config.weight_of(left.value) == config.weight_of(right.value)


@pytest.mark.parametrize(("left", "right"), CONFUSABLE_PAIRS)
def test_confusable_tags_share_a_group(config: BandConfig, left: Tag, right: Tag) -> None:
    """Equal weights only neutralise confusion if both tags feed the same band."""
    assert config.group_of(left.value) == config.group_of(right.value)


def test_no_tag_encodes_wrong_meaning() -> None:
    """Wrong meaning does not live in a span; it is carried by the `meaning` band."""
    assert not {tag for tag in TAGS if "mean" in tag or "sense" in tag}


def test_group_membership_agrees_between_code_and_config(config: BandConfig) -> None:
    """The config supplies numbers; the code owns the taxonomy. They must not drift."""
    for tag in Tag:
        assert config.group_of(tag.value) == GROUP_OF[tag]


def test_group_of_accepts_a_plain_string() -> None:
    assert group_of("agr") is TagGroup.CORRECTNESS
    assert group_of("unnat") is TagGroup.USAGE


def test_group_of_rejects_an_unknown_tag() -> None:
    with pytest.raises(ValueError, match="agreement"):
        group_of("agreement")


def test_every_weight_is_a_positive_integer(config: BandConfig) -> None:
    for tag in Tag:
        weight = config.weight_of(tag.value)
        assert isinstance(weight, int)
        assert weight > 0


# The tests above all spell out `.value`, so they cannot catch a `Tag` that
# stringifies or hashes as something other than its value. Every lookup in the
# band formula is by string, and both defects below fail silently — a wrong
# group, or a missed set membership, with no exception to point at them.


@pytest.mark.parametrize("tag", list(Tag))
def test_a_tag_stringifies_to_its_value(tag: Tag) -> None:
    """`str(Tag.AGR)` must be `agr`, not `Tag.AGR`."""
    assert str(tag) == tag.value
    assert f"{tag}" == tag.value
    assert f"[a>b:{tag}]" == f"[a>b:{tag.value}]"


@pytest.mark.parametrize("tag", list(Tag))
def test_a_tag_hashes_like_its_value(tag: Tag) -> None:
    """Equality and hashing must agree, or set and dict lookups miss silently."""
    assert tag == tag.value
    assert hash(tag) == hash(tag.value)
    assert tag in TAGS
    assert {tag.value: 1}[tag] == 1  # type: ignore[index]


@pytest.mark.parametrize("tag", list(Tag))
def test_group_lookup_accepts_the_enum_itself(config: BandConfig, tag: Tag) -> None:
    """`config.group_of` reads config sets keyed by plain strings."""
    assert config.group_of(tag) == GROUP_OF[tag]
    assert config.weight_of(tag) == config.weight_of(tag.value)


def test_group_stringifies_to_its_value() -> None:
    assert str(TagGroup.CORRECTNESS) == "correctness"
    assert TagGroup.USAGE == "usage"
    assert hash(TagGroup.USAGE) == hash("usage")
