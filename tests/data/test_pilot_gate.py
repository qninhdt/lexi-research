"""The pilot gate: the arithmetic behind the go/no-go decision.

These tests exist because the gate is the one place in the pipeline where a
number decides whether money gets spent. A silently wrong QWK reads as "teacher
is fine" and licenses a 20K-row run on noise, so each metric is checked against a
case whose answer is known by construction rather than by running the code.
"""

from __future__ import annotations

import json

import pytest

from lexi_research.data.pilot_gate import (
    HARD_GATES,
    GateReport,
    GateResult,
    band_distribution,
    edit_f1,
    edit_triples,
    evaluate,
    measure_self_consistency,
    quadratic_weighted_kappa,
)
from lexi_research.format import MAX_BAND, MIN_BAND

# A run that clears every gate. Each test mutates one field, so a failure names
# the gate it broke instead of reporting a report-wide "not passed".
PASSING = {
    "self_consistency_qwk": 0.85,
    "self_consistency_edit_f1": 0.75,
    "meanings": [0, 1, 1, 2, 2, 3, 3, 4],
    "format_validity": 0.97,
    "mean_distinct2": 0.84,
    "other_tag_share": 0.01,
    "batch_single_qwk": 0.9,
}


def report(**overrides: object) -> GateReport:
    payload = dict(PASSING)
    payload.update(overrides)
    return evaluate(**payload)  # type: ignore[arg-type]


def gate(rep: GateReport, name: str) -> GateResult:
    matches = [result for result in rep.results if result.name == name]
    assert matches, f"no gate named {name} in {[r.name for r in rep.results]}"
    return matches[0]


class TestQuadraticWeightedKappa:
    def test_identical_varied_ratings_score_one(self) -> None:
        ratings = [0, 1, 2, 3, 4, 2, 1]
        assert quadratic_weighted_kappa(ratings, ratings) == pytest.approx(1.0)

    def test_constant_and_identical_scores_one(self) -> None:
        """Plain kappa is undefined here; 0.0 would call agreement disagreement.

        Two graders who both said "2" every time agree completely. The
        chance-correction denominator vanishes because neither varied, and the
        formula's fallback has to read that as perfect rather than as noise.
        """
        assert quadratic_weighted_kappa([2, 2, 2, 2], [2, 2, 2, 2]) == pytest.approx(1.0)

    def test_constant_but_disagreeing_scores_zero(self) -> None:
        assert quadratic_weighted_kappa([0, 0, 0], [4, 4, 4]) == pytest.approx(0.0)

    def test_near_misses_beat_total_disagreement(self) -> None:
        """The reason for QWK over accuracy: distance matters on an ordered scale."""
        gold = [0, 1, 2, 3, 4]
        near = [1, 2, 3, 4, 3]
        far = [4, 3, 2, 1, 0]
        assert quadratic_weighted_kappa(gold, near) > quadratic_weighted_kappa(gold, far)

    def test_systematic_disagreement_goes_negative(self) -> None:
        assert quadratic_weighted_kappa([0, 0, 4, 4], [4, 4, 0, 0]) < 0.0

    def test_empty_input_scores_zero(self) -> None:
        """No evidence of agreement is not evidence of agreement — the gate
        must not pass on an empty self-consistency sample."""
        assert quadratic_weighted_kappa([], []) == 0.0

    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="unequal lengths"):
            quadratic_weighted_kappa([1, 2], [1])

    def test_is_symmetric(self) -> None:
        first = [0, 2, 2, 4, 1]
        second = [1, 2, 3, 4, 0]
        assert quadratic_weighted_kappa(first, second) == pytest.approx(
            quadratic_weighted_kappa(second, first)
        )


class TestEditF1:
    def test_two_clean_sentences_agree(self) -> None:
        """Both graders found nothing to fix. Scoring that 0.0 would penalise
        the teacher for the easiest rows in the dataset."""
        assert edit_f1([], []) == pytest.approx(1.0)

    def test_one_empty_side_scores_zero(self) -> None:
        assert edit_f1([(0, 3, "sp")], []) == pytest.approx(0.0)
        assert edit_f1([], [(0, 3, "sp")]) == pytest.approx(0.0)

    def test_identical_edits_score_one(self) -> None:
        edits = [(0, 3, "sp"), (7, 9, "art")]
        assert edit_f1(edits, edits) == pytest.approx(1.0)

    def test_order_does_not_matter(self) -> None:
        """Set semantics: two graders may mark the same edits in either order,
        and that is not a disagreement."""
        assert edit_f1([(7, 9, "art"), (0, 3, "sp")], [(0, 3, "sp"), (7, 9, "art")]) == 1.0

    def test_right_span_wrong_tag_is_a_miss(self) -> None:
        """A tag decides the penalty group, so it decides the band. Matching the
        span while disagreeing on the tag is not partial credit."""
        assert edit_f1([(0, 3, "sp")], [(0, 3, "word")]) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        predicted = [(0, 3, "sp"), (5, 8, "art")]
        gold = [(0, 3, "sp"), (10, 12, "prep")]
        # precision 1/2, recall 1/2 -> F1 1/2
        assert edit_f1(predicted, gold) == pytest.approx(0.5)

    def test_duplicate_predictions_do_not_inflate(self) -> None:
        assert edit_f1([(0, 3, "sp"), (0, 3, "sp")], [(0, 3, "sp")]) == pytest.approx(1.0)


class TestEditTriples:
    def test_null_correction_has_no_edits(self) -> None:
        assert edit_triples(None) == []

    def test_clean_correction_has_no_edits(self) -> None:
        assert edit_triples("I like the book.") == []

    def test_edits_come_back_as_span_tag_triples(self) -> None:
        triples = edit_triples("I [goed>went:form] home.")
        assert triples == [(2, 6, "form")]

    def test_unparseable_correction_contributes_nothing(self) -> None:
        """Two graders who both emitted garbage agree that they failed. Raising
        here would abort a self-consistency measurement over one bad row."""
        assert edit_triples("I [broken home.") == []

    def test_two_graders_who_both_gave_up_agree(self) -> None:
        assert edit_f1(edit_triples(None), edit_triples(None)) == pytest.approx(1.0)


class TestBandDistribution:
    def test_every_band_appears_even_at_zero(self) -> None:
        """G2 asks whether a band is *missing*; a dict that omits absent keys
        makes exactly that condition invisible."""
        distribution = band_distribution([0, 0, 4, 4])
        assert set(distribution) == set(range(MIN_BAND, MAX_BAND + 1))
        assert distribution[1] == 0.0
        assert distribution[2] == 0.0
        assert distribution[3] == 0.0

    def test_shares_sum_to_one(self) -> None:
        distribution = band_distribution([0, 1, 2, 3, 4, 2])
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_shares_are_proportions(self) -> None:
        distribution = band_distribution([2, 2, 2, 4])
        assert distribution[2] == pytest.approx(0.75)
        assert distribution[4] == pytest.approx(0.25)

    def test_empty_input_reports_all_bands_at_zero(self) -> None:
        distribution = band_distribution([])
        assert set(distribution) == set(range(MIN_BAND, MAX_BAND + 1))
        assert all(share == 0.0 for share in distribution.values())


class TestEvaluate:
    def test_a_healthy_run_passes_everything(self) -> None:
        rep = report()
        assert rep.passed
        assert rep.blocking_passed
        assert rep.failures() == ()

    def test_g7_is_absent_unless_a_human_reviewed(self) -> None:
        """A manual review nobody performed must not report as passed."""
        assert "G7_manual_review" not in {result.name for result in report().results}
        assert "G7_manual_review" in {
            result.name for result in report(manual_review_passed=True).results
        }

    def test_a_failed_manual_review_fails_without_blocking(self) -> None:
        rep = report(manual_review_passed=False)
        assert not rep.passed
        assert rep.blocking_passed
        assert gate(rep, "G7_manual_review").value == 0.0

    def test_low_self_consistency_qwk_blocks(self) -> None:
        rep = report(self_consistency_qwk=0.55)
        assert not gate(rep, "G1_self_consistency").passed
        assert not rep.blocking_passed

    def test_low_self_consistency_edit_f1_blocks(self) -> None:
        """G1 has two conditions. A teacher can agree with itself on bands while
        marking different edits every time, and that noise still caps Phase 8."""
        rep = report(self_consistency_edit_f1=0.4)
        assert not gate(rep, "G1_self_consistency").passed
        assert not rep.blocking_passed

    def test_g1_is_inclusive_at_its_thresholds(self) -> None:
        assert gate(
            report(self_consistency_qwk=0.7, self_consistency_edit_f1=0.6),
            "G1_self_consistency",
        ).passed

    def test_a_bimodal_distribution_blocks(self) -> None:
        rep = report(meanings=[0] * 5 + [4] * 5)
        coverage = gate(rep, "G2_band_coverage")
        assert not coverage.passed
        assert not rep.blocking_passed
        assert "1, 2, 3" in coverage.detail or "[1, 2, 3]" in coverage.detail

    def test_a_missing_band_blocks_even_with_a_healthy_middle(self) -> None:
        """The middle share can clear 40% while band 4 never appears at all. A
        dataset with no clean sentences cannot teach the top of the scale."""
        meanings = [0, 1, 1, 2, 2, 3, 3]
        rep = report(meanings=meanings)
        assert band_distribution(meanings)[4] == 0.0
        assert not gate(rep, "G2_band_coverage").passed
        assert not rep.blocking_passed

    def test_a_thin_middle_blocks(self) -> None:
        meanings = [0] * 4 + [1, 2] + [4] * 4  # every band present, middle 20%
        rep = report(meanings=meanings)
        assert not gate(rep, "G2_band_coverage").passed

    def test_g2_is_inclusive_at_forty_percent(self) -> None:
        meanings = [0, 0, 0, 1, 2, 3, 4, 4, 4, 4]  # middle exactly 30%... push to 40%
        meanings = [0, 0, 0, 1, 1, 2, 3, 4, 4, 4]
        rep = report(meanings=meanings)
        assert gate(rep, "G2_band_coverage").value == pytest.approx(0.4)
        assert gate(rep, "G2_band_coverage").passed

    @pytest.mark.parametrize(
        ("field", "value", "name"),
        [
            ("format_validity", 0.5, "G3_format_validity"),
            ("mean_distinct2", 0.3, "G4_batch_diversity"),
            ("other_tag_share", 0.2, "G5_other_tag_share"),
            ("batch_single_qwk", 0.4, "G6_batch_single_parity"),
        ],
    )
    def test_a_soft_gate_warns_without_blocking(self, field: str, value: float, name: str) -> None:
        """G3-G7 are prompt-tuning loops at ~$1 a pass. They must not block
        training on data that has already been paid for."""
        rep = report(**{field: value})
        assert not gate(rep, name).passed
        assert not rep.passed
        assert rep.blocking_passed
        assert [result.name for result in rep.failures()] == [name]

    def test_hard_gates_match_the_blocking_flags(self) -> None:
        """`HARD_GATES` is what Phase 7's entry check asserts on, so it has to
        stay the same set as the gates the report marks blocking."""
        blocking = {result.name for result in report().results if result.blocking}
        assert blocking == set(HARD_GATES)

    def test_every_gate_records_the_threshold_it_was_judged_against(self) -> None:
        for result in report(manual_review_passed=True).results:
            assert result.threshold > 0.0
            assert result.detail, f"{result.name} has no explanation"

    def test_gate_names_are_unique_and_ordered(self) -> None:
        names = [result.name for result in report(manual_review_passed=True).results]
        assert names == sorted(names)
        assert len(names) == len(set(names))


class TestGateReportIO:
    def test_round_trips_through_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = report(manual_review_passed=True)
        path = tmp_path / "reports" / "pilot-gate.json"
        original.write(path)

        loaded = GateReport.read(path)
        assert loaded.passed == original.passed
        assert loaded.blocking_passed == original.blocking_passed
        assert [r.name for r in loaded.results] == [r.name for r in original.results]
        assert [r.blocking for r in loaded.results] == [r.blocking for r in original.results]

    def test_a_failure_survives_the_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "pilot-gate.json"
        report(self_consistency_qwk=0.1).write(path)
        loaded = GateReport.read(path)
        assert not loaded.blocking_passed
        assert [r.name for r in loaded.failures()] == ["G1_self_consistency"]

    def test_the_written_file_carries_no_learner_text(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """`pilot-gate.json` is committed to Git, and everything derived from the
        dictionary stays in DVC. Only aggregates may cross that line."""
        path = tmp_path / "pilot-gate.json"
        report(manual_review_passed=True).write(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert set(payload) == {"passed", "blocking_passed", "gates"}
        for entry in payload["gates"]:
            assert set(entry) == {"name", "value", "threshold", "passed", "blocking", "detail"}
            assert isinstance(entry["value"], (int, float))

    def test_the_file_is_stable_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Sorted keys and a trailing newline: a re-run that changed nothing
        should produce no diff."""
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        report().write(first)
        report().write(second)
        assert first.read_bytes() == second.read_bytes()
        assert first.read_text(encoding="utf-8").endswith("}\n")


class TestMeasureSelfConsistency:
    def test_a_perfectly_consistent_teacher(self) -> None:
        pairs = [
            {
                "meaning": band,
                "meaning_pass2": band,
                "correction": "I [goed>went:form] home.",
                "correction_pass2": "I [goed>went:form] home.",
            }
            for band in (0, 1, 2, 3, 4)
        ]
        qwk, f1 = measure_self_consistency(pairs)
        assert qwk == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_disagreement_shows_up_in_both_numbers(self) -> None:
        pairs = [
            {
                "meaning": 0,
                "meaning_pass2": 4,
                "correction": "I [goed>went:form] home.",
                "correction_pass2": "I goed [home>house:word].",
            },
            {
                "meaning": 4,
                "meaning_pass2": 0,
                "correction": None,
                "correction_pass2": "I [like>enjoy:word] it.",
            },
        ]
        qwk, f1 = measure_self_consistency(pairs)
        assert qwk < 0.0
        assert f1 == pytest.approx(0.0)

    def test_a_null_correction_on_both_passes_counts_as_agreement(self) -> None:
        pairs = [
            {"meaning": 0, "meaning_pass2": 0, "correction": None, "correction_pass2": None},
        ]
        _, f1 = measure_self_consistency(pairs)
        assert f1 == pytest.approx(1.0)

    def test_edit_f1_is_averaged_per_row_not_pooled(self) -> None:
        """Pooling spans across rows would match an edit in row 1 against the
        same offsets in row 2, which are different words entirely."""
        pairs = [
            {
                "meaning": 2,
                "meaning_pass2": 2,
                "correction": "I [goed>went:form] home.",
                "correction_pass2": "I [goed>went:form] home.",
            },
            {
                "meaning": 2,
                "meaning_pass2": 2,
                "correction": "I [goed>went:form] home.",
                "correction_pass2": "I goed home.",
            },
        ]
        _, f1 = measure_self_consistency(pairs)
        assert f1 == pytest.approx(0.5)

    def test_an_empty_sample_scores_zero(self) -> None:
        """An unmeasured teacher is not a consistent teacher: G1 must fail when
        the self-consistency pass produced nothing."""
        assert measure_self_consistency([]) == (0.0, 0.0)
        assert not report(
            self_consistency_qwk=0.0, self_consistency_edit_f1=0.0
        ).blocking_passed
