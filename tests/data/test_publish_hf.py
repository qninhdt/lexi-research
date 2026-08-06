"""What may and may not be published, and whether the card tells the truth.

The upload allowlist is the interesting test surface. Stage A is converted from
W&I+LOCNESS, whose licence forbids redistributing any part of the corpus, so
"`data/gec/` is never uploaded" is a licence obligation rather than a preference —
and an obligation enforced by a directory glob is one nobody notices breaking.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lexi_research.data.publish_hf import (
    FORBIDDEN,
    UPLOAD,
    PublishError,
    build_card,
    resolve_uploads,
)


class TestUploadAllowlist:
    def test_stage_a_is_never_uploaded(self) -> None:
        """The licence obligation, asserted rather than trusted to review."""
        sources = [source for source, _ in UPLOAD]
        assert not any(source.startswith("data/gec/") for source in sources)
        assert not any(source.startswith("data/corpora/") for source in sources)

    def test_the_response_cache_is_never_uploaded(self) -> None:
        """It holds raw provider payloads, including ones the validator rejected."""
        assert not any(source.startswith(".cache/") for source in (s for s, _ in UPLOAD))

    def test_secrets_are_on_the_forbidden_list(self) -> None:
        assert ".env" in FORBIDDEN
        assert "data/gec/" in FORBIDDEN

    def test_every_upload_is_checked_against_the_forbidden_list(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A future edit that adds a forbidden path must fail loudly, not silently."""
        monkeypatch.setattr(
            "lexi_research.data.publish_hf.UPLOAD",
            (("data/gec/train.parquet", "train.parquet"),),
        )
        with pytest.raises(PublishError, match="forbidden"):
            resolve_uploads(tmp_path)

    def test_an_empty_tree_refuses_rather_than_publishing_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(PublishError, match="nothing to upload"):
            resolve_uploads(tmp_path)

    def test_only_existing_files_are_resolved(self, tmp_path: Path) -> None:
        target = tmp_path / "band_config.json"
        target.write_text("{}", encoding="utf-8")

        resolved = resolve_uploads(tmp_path)

        assert [destination for _, destination in resolved] == ["band_config.json"]


@pytest.fixture
def labels_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "raw_labels.parquet"
    rows = [
        {"meaning": 0, "tags": ["art"]},
        {"meaning": 2, "tags": ["agr", "art"]},
        {"meaning": 4, "tags": []},
        {"meaning": 4, "tags": ["other"]},
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def texts_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "raw_texts.parquet"
    pq.write_table(pa.Table.from_pylist([{"text": "a"}, {"text": "b"}]), path)
    return path


def _card(texts: Path, labels: Path, **overrides: object) -> str:
    kwargs: dict = {
        "repo_id": "someone/lexi",
        "texts_path": texts,
        "labels_path": labels,
        "generate_report": {"mean_distinct2": 0.95, "teacher": {"calls": 10}},
        "label_report": {
            "validity_rate": 0.93,
            "middle_band_share": 0.25,
            "teacher": {"calls": 40},
        },
        "sample_report": {"senses": 7},
        "gate": None,
        "ceiling": None,
        "teacher_model": "some-model",
        "teacher_endpoint": "https://example.invalid/v1",
    }
    kwargs.update(overrides)
    return build_card(**kwargs)  # type: ignore[arg-type]


class TestCard:
    def test_counts_come_from_the_parquet_not_the_prose(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        card = _card(texts_parquet, labels_parquet)

        assert "**2** generated learner sentences" in card
        assert "**4** accepted gradings" in card

    def test_every_band_appears_even_at_zero(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        """A band silently omitted is how a coverage hole becomes invisible."""
        card = _card(texts_parquet, labels_parquet)

        for band in range(5):
            assert f"| {band} |" in card
        assert "| 1 | 0 | 0.0% |" in card

    def test_a_failed_blocking_gate_is_stated_prominently(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        """Publishing data that failed its own gate is fine; hiding that is not."""
        gate = {
            "passed": False,
            "blocking_passed": False,
            "gates": [
                {
                    "name": "G2_band_coverage",
                    "value": 0.31,
                    "threshold": 0.4,
                    "passed": False,
                    "blocking": True,
                }
            ],
        }
        card = _card(texts_parquet, labels_parquet, gate=gate)

        assert "**FAIL**" in card
        assert "NOT PASSED" in card
        assert "Read this before training on it" in card

    def test_a_passing_gate_carries_no_warning(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        gate = {
            "passed": True,
            "blocking_passed": True,
            "gates": [
                {
                    "name": "G1_self_consistency",
                    "value": 0.98,
                    "threshold": 0.7,
                    "passed": True,
                    "blocking": True,
                }
            ],
        }
        card = _card(texts_parquet, labels_parquet, gate=gate)

        assert "Read this before training on it" not in card

    def test_the_teacher_model_is_reported_as_the_endpoint_named_it(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        """The string is unverifiable through a proxy, so the card says so."""
        card = _card(texts_parquet, labels_parquet, teacher_model="some-alias")

        assert "`some-alias`" in card
        assert "rather than an independently verified checkpoint" in card

    def test_the_excluded_corpus_is_named(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        """A reader must be able to tell what is missing and why."""
        card = _card(texts_parquet, labels_parquet)

        assert "W&I+LOCNESS" in card
        assert "not" in card.split("## Not included")[1][:200]

    def test_the_no_gold_limitation_survives(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        card = _card(texts_parquet, labels_parquet)

        assert "No human gold set" in card

    def test_the_front_matter_parses_as_yaml_metadata(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        """The Hub reads this block; a malformed one silently loses the metadata."""
        import yaml

        card = _card(texts_parquet, labels_parquet)
        _, front, _ = card.split("---", 2)
        parsed = yaml.safe_load(front)

        assert parsed["language"] == ["en"]
        assert parsed["size_categories"] == ["n<1K"]

    def test_size_category_tracks_the_row_count(self, tmp_path: Path, texts_parquet: Path) -> None:
        big = tmp_path / "big.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"meaning": 3, "tags": []} for _ in range(1200)]), big
        )

        card = _card(texts_parquet, big)

        assert "1K<n<10K" in card

    def test_the_ceiling_section_appears_only_when_measured(
        self, texts_parquet: Path, labels_parquet: Path
    ) -> None:
        without = _card(texts_parquet, labels_parquet)
        assert "self-consistency (the ceiling" not in without

        with_ceiling = _card(
            texts_parquet,
            labels_parquet,
            ceiling={"meaning_qwk": 0.98, "correction_edit_f1": 0.85, "sampled": 194},
        )
        assert "self-consistency (the ceiling" in with_ceiling
        assert "0.98" in with_ceiling


def test_a_missing_report_is_a_typed_error(tmp_path: Path, monkeypatch) -> None:
    """Publishing must not invent a number the pipeline never wrote."""
    from lexi_research.data.publish_hf import _read_json

    monkeypatch.chdir(tmp_path)
    with pytest.raises(PublishError, match="missing"):
        _read_json("data/raw/label-report.json")


def test_the_real_upload_list_names_no_directory(tmp_path: Path) -> None:
    """Every entry is one file, so a new file cannot ride along by being adjacent."""
    for source, _ in UPLOAD:
        assert not source.endswith("/")
        assert Path(source).suffix in {".parquet", ".json"}
