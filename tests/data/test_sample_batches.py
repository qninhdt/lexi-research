"""Sampler tests.

Two properties do the real work. Determinism: the same seed and pool must yield
the same specs, because the resume cache is keyed on spec identity and a reshuffle
would re-spend the whole budget. And distinctness within a batch: K specs sharing
a sense must occupy different grid cells, which is the structural half of the
diversity defence.
"""

from __future__ import annotations

import pytest

from lexi_research.data.profiles import ProfileRegistry, load_profiles
from lexi_research.data.sample_batches import (
    ERROR_SPECS,
    MEANING_REQS,
    SPEC_COLUMNS,
    K,
    Sense,
    build_batches,
    load_weights,
    sample_senses,
    spec_rows,
    spec_uid,
    write_specs,
)


def _sense(index: int, *, multiword: bool = False) -> Sense:
    target = f"word{index}" if not multiword else f"give up{index}"
    return Sense(
        sense_uid=f"{index:016x}",
        target=target,
        target_norm=target,
        pos="noun" if not multiword else "phrasal verb",
        definition=f"definition number {index}",
        cefr="B1" if index % 3 else None,
        is_multiword=multiword,
        is_placeholder=False,
    )


@pytest.fixture(scope="module")
def registry() -> ProfileRegistry:
    return load_profiles()


@pytest.fixture(scope="module")
def pool() -> list[Sense]:
    return [_sense(i) for i in range(80)] + [_sense(100 + i, multiword=True) for i in range(20)]


class TestSpecUid:
    def test_is_stable(self) -> None:
        first = spec_uid("abc", 2, "few", "vi-b1-tense", 7)
        assert first == spec_uid("abc", 2, "few", "vi-b1-tense", 7)

    @pytest.mark.parametrize(
        "changed",
        [
            ("xyz", 2, "few", "vi-b1-tense", 7),
            ("abc", 3, "few", "vi-b1-tense", 7),
            ("abc", 2, "many", "vi-b1-tense", 7),
            ("abc", 2, "few", "ja-b1-prepositions", 7),
            ("abc", 2, "few", "vi-b1-tense", 8),
        ],
    )
    def test_every_component_participates(self, changed: tuple) -> None:
        """A component missing from the hash would collide two different requests."""
        assert spec_uid("abc", 2, "few", "vi-b1-tense", 7) != spec_uid(*changed)


class TestSampleSenses:
    def test_returns_the_requested_count(self, pool: list[Sense]) -> None:
        assert len(sample_senses(pool, 30, seed=1)) == 30

    def test_is_deterministic(self, pool: list[Sense]) -> None:
        first = [s.sense_uid for s in sample_senses(pool, 30, seed=1)]
        second = [s.sense_uid for s in sample_senses(pool, 30, seed=1)]
        assert first == second

    def test_seed_changes_the_sample(self, pool: list[Sense]) -> None:
        first = {s.sense_uid for s in sample_senses(pool, 30, seed=1)}
        second = {s.sense_uid for s in sample_senses(pool, 30, seed=2)}
        assert first != second

    def test_does_not_depend_on_input_order(self, pool: list[Sense]) -> None:
        """Parquet row order must not leak into the sample."""
        forward = [s.sense_uid for s in sample_senses(pool, 30, seed=1)]
        backward = [s.sense_uid for s in sample_senses(list(reversed(pool)), 30, seed=1)]
        assert forward == backward

    def test_honours_the_multiword_share(self, pool: list[Sense]) -> None:
        picked = sample_senses(pool, 40, seed=1, multiword_share=0.25)
        multiword = sum(1 for s in picked if s.is_multiword)
        assert multiword == 10

    def test_returns_no_duplicates(self, pool: list[Sense]) -> None:
        picked = sample_senses(pool, 50, seed=3)
        assert len({s.sense_uid for s in picked}) == len(picked)

    def test_zero_count_is_empty(self, pool: list[Sense]) -> None:
        assert sample_senses(pool, 0, seed=1) == []

    def test_a_pool_short_on_multiword_still_fills_the_count(self) -> None:
        """Falling short on one kind spends the remainder on the other."""
        thin = [_sense(i) for i in range(10)] + [_sense(99, multiword=True)]
        picked = sample_senses(thin, 8, seed=1, multiword_share=0.5)
        assert len(picked) == 8

    def test_a_smaller_draw_is_a_prefix_of_a_larger_one(self, pool: list[Sense]) -> None:
        """The property that makes growing a dataset affordable.

        Generation is paid per sense. If raising the count re-drew a fresh
        combination, the senses already generated would mostly fall out of the
        sample and their cost would be thrown away. Nesting means a larger run
        re-requests the same senses — whose `batch_uid`s are unchanged — so the
        response cache serves them and only new senses reach the network.
        """
        small = {s.sense_uid for s in sample_senses(pool, 20, seed=1)}
        medium = {s.sense_uid for s in sample_senses(pool, 40, seed=1)}
        large = {s.sense_uid for s in sample_senses(pool, 60, seed=1)}

        assert small <= medium <= large

    def test_growing_the_count_keeps_every_batch_uid(
        self, pool: list[Sense], registry: ProfileRegistry
    ) -> None:
        """Nesting is only worth anything if the cache key survives it too."""
        before, _ = build_batches(sample_senses(pool, 20, seed=1), registry, seed=1)
        after, _ = build_batches(sample_senses(pool, 50, seed=1), registry, seed=1)

        uids_after = {batch.sense.sense_uid: batch.batch_uid for batch in after}
        assert all(batch.sense.sense_uid in uids_after for batch in before)
        assert all(uids_after[batch.sense.sense_uid] == batch.batch_uid for batch in before)

    def test_a_grown_pool_does_not_reshuffle_the_existing_draw(
        self, pool: list[Sense]
    ) -> None:
        """Ranking is per-sense, so new senses interleave without displacing paid work.

        They may push a sense past the cut when the count is fixed, but they must
        never reorder the senses that remain — that would change which ones a
        resumed run considers already done.
        """
        before = [s.sense_uid for s in sample_senses(pool, 30, seed=1)]
        grown = list(pool) + [_sense(10_000 + i) for i in range(25)]
        after = [s.sense_uid for s in sample_senses(grown, 30, seed=1)]

        survivors = [uid for uid in after if uid in set(before)]
        assert survivors == [uid for uid in before if uid in set(after)]


class TestBuildBatches:
    def test_one_batch_per_sense(self, registry: ProfileRegistry) -> None:
        senses = [_sense(i) for i in range(5)]
        batches, stats = build_batches(senses, registry, seed=1)
        assert len(batches) == 5
        assert stats.senses == 5

    def test_k_specs_per_batch(self, registry: ProfileRegistry) -> None:
        batches, stats = build_batches([_sense(1)], registry, seed=1)
        assert len(batches[0].specs) == K
        assert stats.specs == K

    def test_cells_inside_a_batch_are_distinct(self, registry: ProfileRegistry) -> None:
        """Two identical cells in one call ask for the same sentence twice."""
        batches, _ = build_batches([_sense(i) for i in range(20)], registry, seed=1)
        for batch in batches:
            cells = [(spec.meaning_req, spec.error_spec) for spec in batch.specs]
            assert len(set(cells)) == len(cells), batch.batch_uid

    def test_spec_ids_are_unique_within_a_batch(self, registry: ProfileRegistry) -> None:
        batches, _ = build_batches([_sense(i) for i in range(20)], registry, seed=1)
        for batch in batches:
            ids = [spec.spec_id for spec in batch.specs]
            assert len(set(ids)) == len(ids)

    def test_is_deterministic(self, registry: ProfileRegistry) -> None:
        senses = [_sense(i) for i in range(10)]
        first, _ = build_batches(senses, registry, seed=5)
        second, _ = build_batches(senses, registry, seed=5)
        assert [b.batch_uid for b in first] == [b.batch_uid for b in second]
        assert [s.spec_id for b in first for s in b.specs] == [
            s.spec_id for b in second for s in b.specs
        ]

    def test_adding_a_sense_does_not_reshuffle_the_others(
        self, registry: ProfileRegistry
    ) -> None:
        """Per-sense RNG streams: a grown pool must not invalidate a warm cache."""
        senses = [_sense(i) for i in range(10)]
        before, _ = build_batches(senses, registry, seed=5)
        after, _ = build_batches([*senses, _sense(999)], registry, seed=5)
        assert [b.batch_uid for b in before] == [b.batch_uid for b in after[:10]]

    def test_seed_changes_the_specs(self, registry: ProfileRegistry) -> None:
        senses = [_sense(i) for i in range(10)]
        first, _ = build_batches(senses, registry, seed=1)
        second, _ = build_batches(senses, registry, seed=2)
        assert [b.batch_uid for b in first] != [b.batch_uid for b in second]

    def test_weights_bias_the_middle_bands(self, registry: ProfileRegistry) -> None:
        """The whole point of the weights: bands 1-3 must dominate."""
        senses = [_sense(i) for i in range(200)]
        weights = {0: 1.0, 1: 3.0, 2: 3.0, 3: 3.0, 4: 2.0}
        _, stats = build_batches(senses, registry, seed=1, meaning_weights=weights)
        total = sum(stats.meaning_req_counts.values())
        middle = sum(stats.meaning_req_counts.get(band, 0) for band in (1, 2, 3))
        assert middle / total > 0.40

    def test_every_axis_value_appears_across_a_large_sample(
        self, registry: ProfileRegistry
    ) -> None:
        senses = [_sense(i) for i in range(200)]
        _, stats = build_batches(senses, registry, seed=1)
        assert set(stats.meaning_req_counts) == set(MEANING_REQS)
        assert set(stats.error_spec_counts) == set(ERROR_SPECS)

    def test_near_native_profiles_do_not_get_low_meaning_cells(
        self, registry: ProfileRegistry
    ) -> None:
        """Asking a fluent writer for meaning 0 yields a contradiction, not a row."""
        near_native = {p.id for p in registry.near_native}
        senses = [_sense(i) for i in range(100)]
        batches, _ = build_batches(senses, registry, seed=1)
        for batch in batches:
            for spec in batch.specs:
                if spec.profile_id in near_native:
                    assert spec.meaning_req >= 3, spec

    def test_error_bias_comes_from_the_profile(self, registry: ProfileRegistry) -> None:
        batches, _ = build_batches([_sense(1)], registry, seed=1)
        for spec in batches[0].specs:
            assert spec.error_bias == registry.by_id(spec.profile_id).error_bias


class TestSpecRows:
    def test_columns_match_the_contract(self, registry: ProfileRegistry) -> None:
        batches, _ = build_batches([_sense(1)], registry, seed=1)
        rows = list(spec_rows(batches, 1))
        assert set(rows[0]) == set(SPEC_COLUMNS)

    def test_one_row_per_spec(self, registry: ProfileRegistry) -> None:
        batches, stats = build_batches([_sense(i) for i in range(4)], registry, seed=1)
        assert len(list(spec_rows(batches, 1))) == stats.specs

    def test_carries_the_sense_flags(self, registry: ProfileRegistry) -> None:
        batches, _ = build_batches([_sense(1, multiword=True)], registry, seed=1)
        rows = list(spec_rows(batches, 1))
        assert all(row["is_multiword"] for row in rows)


class TestWriteSpecs:
    def test_writes_every_row(self, registry: ProfileRegistry, tmp_path) -> None:
        batches, stats = build_batches([_sense(i) for i in range(3)], registry, seed=1)
        count = write_specs(batches, 1, tmp_path / "specs.parquet")
        assert count == stats.specs

    def test_schema_is_pinned(self, registry: ProfileRegistry, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        batches, _ = build_batches([_sense(1)], registry, seed=1)
        path = tmp_path / "specs.parquet"
        write_specs(batches, 1, path)
        schema = pq.read_schema(path)
        assert schema.names == list(SPEC_COLUMNS)
        assert schema.field("meaning_req").type == pa.int32()

    def test_two_writes_are_byte_identical(self, registry: ProfileRegistry, tmp_path) -> None:
        from lexi_research.data import sha256_file

        batches, _ = build_batches([_sense(i) for i in range(3)], registry, seed=1)
        first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
        write_specs(batches, 1, first)
        write_specs(batches, 1, second)
        assert sha256_file(first) == sha256_file(second)


class TestLoadWeights:
    def test_coerces_yaml_string_keys(self) -> None:
        meaning, error = load_weights(
            {"meaning_weights": {"0": 1, "3": 3}, "error_spec_weights": {"none": 2}}
        )
        assert meaning == {0: 1.0, 3: 3.0}
        assert error == {"none": 2.0}

    def test_missing_blocks_fall_back_to_uniform(self) -> None:
        meaning, error = load_weights({})
        assert set(meaning) == set(MEANING_REQS)
        assert set(error) == set(ERROR_SPECS)
