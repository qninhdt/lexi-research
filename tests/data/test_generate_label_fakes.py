"""Call 1 and call 2 against a fake teacher — no network, no spend.

The behaviours under test are the ones that cost money when they break: resume
after a crash must not re-pay for completed work, one bad element must not
discard its five good siblings, and a batch whose element count comes back wrong
must fall back to singletons rather than being thrown away.

The last test is structural rather than behavioural: it asserts that no spec
column reaches a label row. That is the distillation-validity property the whole
two-call design exists to protect, and it is exactly the kind of thing a later
refactor breaks silently.
"""

from __future__ import annotations

import pytest

from lexi_research.data.generate import (
    TEXT_COLUMNS,
    generate_batches,
    target_is_present,
    text_rows,
    validate_text,
    write_texts,
)
from lexi_research.data.jsonl_store import JsonlStore
from lexi_research.data.label import (
    LABEL_COLUMNS,
    label_texts,
    label_rows,
    read_texts_for_labelling,
    self_consistency_pairs,
    write_labels,
)
from lexi_research.data.profiles import load_profiles
from lexi_research.data.sample_batches import Sense, build_batches
from lexi_research.teacher import (
    DiversifyBatch,
    GraderOutput,
    NullCache,
    TeacherClient,
    TeacherConfig,
    prompt_hash,
)


def _config(**overrides) -> TeacherConfig:
    base = {
        "base_url": "http://fake",
        "api_key": "k",
        "model": "fake-teacher",
        "concurrency": 4,
        "max_retries": 2,
        "base_delay": 0.0,
    }
    base.update(overrides)
    return TeacherConfig(**base)


def _sense(index: int = 1) -> Sense:
    return Sense(
        sense_uid=f"{index:016x}",
        target="bright",
        target_norm="bright",
        pos="adjective",
        definition="full of light",
        cefr="A2",
        is_multiword=False,
        is_placeholder=False,
    )


class FakeDiversifier:
    """Returns one distinct sentence per requested spec.

    Sentences vary by spec so a diversity measurement over the batch is
    meaningful; `script` lets a test override the reply for one call.
    """

    def __init__(self, *, script: list | None = None) -> None:
        self.calls = 0
        self.script = script or []
        self.last_usage = (10, 20)

    async def parse(self, messages, schema):
        self.calls += 1
        if self.script:
            reply = self.script.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return schema.model_validate(reply)

        spec_ids = _spec_ids_from(messages)
        return DiversifyBatch.model_validate(
            {
                "sentences": [
                    {
                        "spec_id": spec_id,
                        "text": f"The {noun} was bright and {adjective} yesterday.",
                    }
                    for spec_id, noun, adjective in zip(
                        spec_ids,
                        ["room", "garden", "kitchen", "hallway", "office", "cellar"],
                        ["airy", "warm", "quiet", "narrow", "busy", "damp"],
                    )
                ]
            }
        )


def _spec_ids_from(messages) -> list[str]:
    """Pull the spec ids back out of the rendered call-1 prompt."""
    user = messages[-1]["content"]
    return [
        line.split("spec_id:", 1)[1].strip()
        for line in user.splitlines()
        if "spec_id:" in line
    ]


class FakeGrader:
    """Grades any text as a clean sentence, echoing it back verbatim.

    Echoing matters: check 3 requires the stripped correction to equal the input
    exactly, so a fake that returned a fixed string would fail validation for
    reasons unrelated to what a test is asserting.
    """

    def __init__(self, *, script: list | None = None, meaning: int = 4) -> None:
        self.calls = 0
        self.texts: list[str] = []
        self.script = script or []
        self.meaning = meaning
        self.last_usage = (30, 15)

    async def parse(self, messages, schema):
        self.calls += 1
        if self.script:
            reply = self.script.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return schema.model_validate(reply)

        text = _text_from(messages)
        self.texts.append(text)
        return GraderOutput.model_validate(
            {"correction": text, "meaning": self.meaning, "feedback": "Good use of the word."}
        )


def _text_from(messages) -> str:
    """Recover the learner sentence from inside the untrusted block."""
    user = messages[-1]["content"]
    start = user.index("<untrusted-")
    start = user.index("\n", start) + 1
    end = user.rindex("</untrusted-")
    return user[start:end].strip()


def _client(llm) -> TeacherClient:
    return TeacherClient(_config(), cache=NullCache(), llm=llm, prompt_hash=prompt_hash())


@pytest.fixture
def batches():
    registry = load_profiles()
    built, _ = build_batches([_sense(i) for i in range(3)], registry, seed=1)
    return built, registry


class TestValidateText:
    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("", "empty"),
            ("   ", "empty"),
            ("ab", "too_short"),
            ("x" * 601, "too_long"),
            ("two\nlines", "multiline"),
        ],
    )
    def test_rejects_malformed(self, text: str, reason: str) -> None:
        assert validate_text(text) == reason

    def test_accepts_a_plausible_sentence(self) -> None:
        assert validate_text("The room was bright.") is None


class TestTargetPresence:
    @pytest.mark.parametrize(
        "text",
        [
            "The room was bright.",
            "She smiled brightly at me.",
            "It is the brightest lamp.",
        ],
    )
    def test_inflections_count(self, text: str) -> None:
        """A strict match would reject exactly the realistic learner attempts."""
        assert target_is_present(text, "bright")

    def test_absent_target_is_detected(self) -> None:
        assert not target_is_present("The room was very nice.", "bright")

    def test_placeholders_are_not_required(self) -> None:
        assert target_is_present("She put him down.", "put sb down")


class TestGenerate:
    async def test_produces_one_row_per_spec(self, batches, tmp_path) -> None:
        built, registry = batches
        store = JsonlStore(tmp_path / "texts.jsonl")
        stats = await generate_batches(
            built, _client(FakeDiversifier()), store, traits=registry.traits_map()
        )
        assert stats.batches == len(built)
        assert stats.texts == sum(len(b.specs) for b in built)

    async def test_rows_carry_the_full_contract(self, batches, tmp_path) -> None:
        built, registry = batches
        store = JsonlStore(tmp_path / "texts.jsonl")
        await generate_batches(
            built, _client(FakeDiversifier()), store, traits=registry.traits_map()
        )
        rows = text_rows(store)
        assert set(rows[0]) >= set(TEXT_COLUMNS)

    async def test_resume_skips_completed_batches(self, batches, tmp_path) -> None:
        """The property that makes an interrupted run cheap to restart."""
        built, registry = batches
        store = JsonlStore(tmp_path / "texts.jsonl")
        first = FakeDiversifier()
        await generate_batches(built, _client(first), store, traits=registry.traits_map())

        second = FakeDiversifier()
        stats = await generate_batches(
            built, _client(second), store, traits=registry.traits_map()
        )
        assert second.calls == 0
        assert stats.batches_cached == len(built)
        assert stats.batches == 0

    async def test_partial_run_resumes_the_remainder(self, batches, tmp_path) -> None:
        built, registry = batches
        store = JsonlStore(tmp_path / "texts.jsonl")
        await generate_batches(
            built[:1], _client(FakeDiversifier()), store, traits=registry.traits_map()
        )

        second = FakeDiversifier()
        stats = await generate_batches(
            built, _client(second), store, traits=registry.traits_map()
        )
        assert second.calls == len(built) - 1
        assert stats.batches_cached == 1

    async def test_one_bad_element_does_not_lose_the_batch(self, batches, tmp_path) -> None:
        built, registry = batches
        spec_ids = [spec.spec_id for spec in built[0].specs]
        script = [
            {
                "sentences": [
                    {"spec_id": spec_ids[0], "text": ""},  # rejected
                    *[
                        {"spec_id": sid, "text": f"The room was bright, number {i}."}
                        for i, sid in enumerate(spec_ids[1:])
                    ],
                ]
            }
        ]
        store = JsonlStore(tmp_path / "texts.jsonl")
        stats = await generate_batches(
            built[:1],
            _client(FakeDiversifier(script=script)),
            store,
            traits=registry.traits_map(),
        )
        assert stats.texts == len(spec_ids) - 1
        assert stats.reject_reasons.get("empty") == 1

    async def test_missing_spec_id_is_counted(self, batches, tmp_path) -> None:
        built, registry = batches
        spec_ids = [spec.spec_id for spec in built[0].specs]
        script = [{"sentences": [{"spec_id": spec_ids[0], "text": "The room was bright."}]}]
        store = JsonlStore(tmp_path / "texts.jsonl")
        stats = await generate_batches(
            built[:1],
            _client(FakeDiversifier(script=script)),
            store,
            traits=registry.traits_map(),
        )
        assert stats.reject_reasons.get("missing_spec_id") == len(spec_ids) - 1

    async def test_a_failed_batch_is_counted_not_raised(self, batches, tmp_path) -> None:
        built, registry = batches
        script = [RuntimeError("boom"), RuntimeError("boom")]
        store = JsonlStore(tmp_path / "texts.jsonl")
        stats = await generate_batches(
            built[:1],
            _client(FakeDiversifier(script=script)),
            store,
            traits=registry.traits_map(),
        )
        assert stats.batches_failed == 1
        assert stats.texts == 0

    async def test_a_failed_batch_is_retried_on_the_next_run(self, batches, tmp_path) -> None:
        """No `batch_done` marker means the work is genuinely re-attempted."""
        built, registry = batches
        store = JsonlStore(tmp_path / "texts.jsonl")
        await generate_batches(
            built[:1],
            _client(FakeDiversifier(script=[RuntimeError("x"), RuntimeError("x")])),
            store,
            traits=registry.traits_map(),
        )
        second = FakeDiversifier()
        stats = await generate_batches(
            built[:1], _client(second), store, traits=registry.traits_map()
        )
        assert second.calls == 1
        assert stats.texts > 0

    async def test_collapsed_batches_are_flagged_not_dropped(self, batches, tmp_path) -> None:
        built, registry = batches
        spec_ids = [spec.spec_id for spec in built[0].specs]
        same = "The room was bright and airy today."
        script = [{"sentences": [{"spec_id": sid, "text": same} for sid in spec_ids]}]
        store = JsonlStore(tmp_path / "texts.jsonl")
        stats = await generate_batches(
            built[:1],
            _client(FakeDiversifier(script=script)),
            store,
            traits=registry.traits_map(),
        )
        assert stats.low_diversity_batches == 1
        assert stats.texts == len(spec_ids)  # kept: the number informs the prompt

    async def test_write_texts_pins_the_schema(self, batches, tmp_path) -> None:
        import pyarrow.parquet as pq

        built, registry = batches
        store = JsonlStore(tmp_path / "texts.jsonl")
        await generate_batches(
            built, _client(FakeDiversifier()), store, traits=registry.traits_map()
        )
        path = tmp_path / "raw_texts.parquet"
        write_texts(text_rows(store), path)
        assert pq.read_schema(path).names == list(TEXT_COLUMNS)


async def _texts(built, registry, tmp_path, name: str = "texts.jsonl"):
    store = JsonlStore(tmp_path / name)
    await generate_batches(
        built, _client(FakeDiversifier()), store, traits=registry.traits_map()
    )
    return text_rows(store)


class TestLabel:
    async def test_labels_every_text(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        store = JsonlStore(tmp_path / "labels.jsonl")
        stats = await label_texts(texts, _client(FakeGrader()), store)
        assert stats.labelled == len(texts)
        assert stats.rejected == 0

    async def test_resume_skips_labelled_texts(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts, _client(FakeGrader()), store)

        second = FakeGrader()
        stats = await label_texts(texts, _client(second), store)
        assert second.calls == 0
        assert stats.cached == len(texts)

    async def test_bands_are_derived_not_taken_from_the_model(self, batches, tmp_path) -> None:
        """`grammar` and `naturalness` come from the formula, never from the payload."""
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts, _client(FakeGrader()), store)
        rows = label_rows(store)
        assert all(row["grammar"] == 4 for row in rows)  # clean corrections
        assert all(row["naturalness"] == 4 for row in rows)

    async def test_a_text_altering_correction_is_rejected(self, batches, tmp_path) -> None:
        """Check 3: the model must not reword text it did not mark."""
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        script = [
            {
                "correction": "A completely different sentence.",
                "meaning": 4,
                "feedback": "Fine.",
            }
        ]
        store = JsonlStore(tmp_path / "labels.jsonl")
        stats = await label_texts(texts[:1], _client(FakeGrader(script=script)), store)
        assert stats.rejected == 1
        assert stats.reject_reasons.get("text_altered") == 1

    async def test_an_out_of_range_meaning_is_rejected(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        script = [{"correction": texts[0]["text"], "meaning": 9, "feedback": "Fine."}]
        store = JsonlStore(tmp_path / "labels.jsonl")
        stats = await label_texts(texts[:1], _client(FakeGrader(script=script)), store)
        assert stats.reject_reasons.get("meaning_range") == 1

    async def test_a_null_correction_scores_grammar_zero(self, batches, tmp_path) -> None:
        """The inverted-failure guard: unreadable must not score 4."""
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        script = [{"correction": None, "meaning": 0, "feedback": "I cannot read this."}]
        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts[:1], _client(FakeGrader(script=script)), store)
        row = label_rows(store)[0]
        assert row["grammar"] == 0
        assert row["correction"] is None

    async def test_one_rejected_label_does_not_stop_the_rest(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        script = [{"correction": "wrong text", "meaning": 4, "feedback": "Fine."}]
        store = JsonlStore(tmp_path / "labels.jsonl")
        grader = FakeGrader(script=script)
        stats = await label_texts(texts, _client(grader), store)
        assert stats.rejected == 1
        assert stats.labelled == len(texts) - 1

    async def test_tags_are_extracted_from_the_correction(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        original = texts[0]["text"]
        marked = original.replace("was", "[was>were:agr]", 1)
        script = [{"correction": marked, "meaning": 3, "feedback": "Check the verb."}]
        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts[:1], _client(FakeGrader(script=script)), store)
        row = label_rows(store)[0]
        assert row["tags"] == ["agr"]
        assert row["n_edits"] == 1

    async def test_write_labels_pins_the_schema(self, batches, tmp_path) -> None:
        import pyarrow.parquet as pq

        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts, _client(FakeGrader()), store)
        path = tmp_path / "raw_labels.parquet"
        write_labels(label_rows(store), path)
        assert pq.read_schema(path).names == list(LABEL_COLUMNS)


class TestSpecIsolation:
    async def test_no_spec_column_reaches_a_label_row(self, batches, tmp_path) -> None:
        """The distillation-validity invariant: call 2 never sees the spec.

        `meaning_req` and `error_spec` are diagnostics carried alongside the
        dataset, never inputs to the label. If either appeared in a label row, a
        later join could silently train the student on the instruction rather
        than on the teacher's reading of the text.
        """
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts, _client(FakeGrader()), store)

        for row in label_rows(store):
            assert "meaning_req" not in row
            assert "error_spec" not in row
            assert "profile_id" not in row

    async def test_the_grader_prompt_never_mentions_the_spec(self, batches, tmp_path) -> None:
        """Measured at the wire, not asserted about the code."""
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)

        seen: list[str] = []

        class Recorder(FakeGrader):
            async def parse(self, messages, schema):
                seen.append(" ".join(msg["content"] for msg in messages))
                return await super().parse(messages, schema)

        store = JsonlStore(tmp_path / "labels.jsonl")
        await label_texts(texts, _client(Recorder()), store)

        for prompt in seen:
            assert "meaning_req" not in prompt
            assert "error_spec" not in prompt
            for row in texts:
                assert row["profile_id"] not in prompt


class TestSelfConsistency:
    async def test_regrades_a_shuffled_sample(self, batches, tmp_path) -> None:
        """Gate G1 needs the teacher to answer again, so the cache is bypassed."""
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        grader = FakeGrader()
        labels = [
            {**text, "meaning": 4, "correction": text["text"]}
            for text in texts
        ]
        pairs = await self_consistency_pairs(labels, _client(grader), sample=5, seed=1)
        assert len(pairs) == 5
        assert grader.calls == 5

    async def test_order_differs_from_the_input(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        labels = [{**text, "meaning": 4, "correction": text["text"]} for text in texts]
        pairs = await self_consistency_pairs(labels, _client(FakeGrader()), sample=10, seed=1)
        assert [pair["req_uid"] for pair in pairs] != [row["req_uid"] for row in texts[:10]]

    async def test_is_deterministic_for_a_seed(self, batches, tmp_path) -> None:
        built, registry = batches
        texts = await _texts(built, registry, tmp_path)
        labels = [{**text, "meaning": 4, "correction": text["text"]} for text in texts]
        first = await self_consistency_pairs(labels, _client(FakeGrader()), sample=5, seed=3)
        second = await self_consistency_pairs(labels, _client(FakeGrader()), sample=5, seed=3)
        assert [p["req_uid"] for p in first] == [p["req_uid"] for p in second]


class TestReadTexts:
    def test_round_trips_through_parquet(self, tmp_path) -> None:
        rows = [
            {name: None for name in TEXT_COLUMNS}
            | {
                "req_uid": "a" * 16,
                "batch_uid": "b" * 16,
                "spec_id": "a" * 16,
                "sense_uid": "c" * 16,
                "target": "bright",
                "target_norm": "bright",
                "pos": "adjective",
                "definition": "full of light",
                "cefr": "A2",
                "is_multiword": False,
                "is_placeholder": False,
                "profile_id": "vi-b1-tense",
                "meaning_req": 3,
                "error_spec": "one",
                "text": "The room was bright.",
            }
        ]
        path = tmp_path / "raw_texts.parquet"
        write_texts(rows, path)
        assert read_texts_for_labelling(path)[0]["text"] == "The room was bright."
