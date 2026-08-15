"""Stage-A prompt and collator invariants."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lexi_research.train.collate_corrector import (
    build_corrector_example,
    corrector_answer,
    corrector_messages,
)
from lexi_research.train.corrector_prompt import (
    CORRECTOR_PROMPTS_DIR,
    CORRECTOR_TEMPLATES,
    corrector_prompt_hash,
    render_corrector_prompt,
)


def _stub_tokenizer():
    path = Path(__file__).with_name("test_collate.py")
    spec = importlib.util.spec_from_file_location("_stage_b_collate_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_stage_b_collate_tests"] = module
    spec.loader.exec_module(module)
    return module.StubTokenizer()


ROW = {"text": "He speak well.", "correction": "He [speak>speaks:agr] well."}


class TestPromptIsolation:
    def test_stage_a_templates_do_not_change_the_teacher_hash(self) -> None:
        """Stage-A prompts must not live under `teacher/prompts/`."""
        from lexi_research.teacher.registry import PROMPTS_DIR, template_names

        on_disk = set(template_names())
        for name in set(CORRECTOR_TEMPLATES):
            assert name not in on_disk
            assert not (PROMPTS_DIR / name).exists()

    def test_the_two_prompt_hashes_are_independent(self) -> None:
        from lexi_research.teacher.registry import prompt_hash

        assert corrector_prompt_hash() != prompt_hash()

    def test_the_hash_covers_every_template_on_disk(self) -> None:
        on_disk = {path.name for path in CORRECTOR_PROMPTS_DIR.glob("*.jinja")}
        assert on_disk == set(CORRECTOR_TEMPLATES)

    def test_the_hash_changes_when_a_template_changes(self, tmp_path, monkeypatch) -> None:
        from lexi_research.train import corrector_prompt

        before = corrector_prompt_hash()
        staging = tmp_path / "prompts"
        staging.mkdir()
        for path in CORRECTOR_PROMPTS_DIR.glob("*.jinja"):
            (staging / path.name).write_bytes(path.read_bytes())
        target = staging / "corrector_system.jinja"
        target.write_text(target.read_text(encoding="utf-8") + "\nExtra.\n", encoding="utf-8")

        monkeypatch.setattr(corrector_prompt, "CORRECTOR_PROMPTS_DIR", staging)
        assert corrector_prompt.corrector_prompt_hash() != before


    def test_it_requests_minimal_edits(self) -> None:
        system = render_corrector_prompt("x")[0]["content"].lower()
        assert "minimal edits" in system

    def test_it_does_not_ask_for_meaning_or_feedback(self) -> None:
        """Stage A has labels for neither; asking would invite invented answers."""
        system = render_corrector_prompt("x")[0]["content"]
        assert '"meaning"' not in system
        assert '"feedback"' not in system

    def test_it_does_not_ask_for_derived_bands(self) -> None:
        system = render_corrector_prompt("x")[0]["content"]
        assert '"grammar"' not in system
        assert '"naturalness"' not in system

    def test_it_is_shorter_than_the_grader_prompt(self) -> None:
        from lexi_research.teacher import render_grader_prompt
        from lexi_research.teacher.schemas import SenseRef

        mine = sum(len(m["content"]) for m in render_corrector_prompt("x"))
        theirs = sum(
            len(m["content"])
            for m in render_grader_prompt(
                "bright", SenseRef(definition="full of light", pos="adjective"), "x", nonce="0"
            )
        )
        assert mine < theirs

    def test_render_is_deterministic(self) -> None:
        assert render_corrector_prompt("hi") == render_corrector_prompt("hi")

    def test_user_contains_learner_text(self) -> None:
        user = render_corrector_prompt("He speak.")[1]["content"]
        assert "Sentence: He speak." in user

    def test_a_missing_variable_raises_rather_than_rendering_empty(self) -> None:
        from jinja2 import UndefinedError

        from lexi_research.train import corrector_prompt

        template = corrector_prompt._ENV.from_string("{{ nope }}")
        with pytest.raises(UndefinedError):
            template.render()


class TestCollator:
    def test_the_answer_is_the_corrected_sentence(self) -> None:
        assert corrector_answer(ROW) == "He speaks well."

    def test_an_unreadable_sentence_is_spelled_null(self) -> None:
        assert corrector_answer({"correction": None}) == "null"

    def test_a_non_string_correction_raises(self) -> None:
        from lexi_research.train.collate import CollationError

        with pytest.raises(CollationError):
            corrector_answer({"correction": 7})

    def test_only_the_answer_is_supervised(self) -> None:
        from lexi_research.train.collate import IGNORE_INDEX

        example = build_corrector_example(_stub_tokenizer(), ROW)
        assert example.supervised_tokens > 0
        assert example.supervised_tokens < len(example.input_ids)
        assert example.labels[0] == IGNORE_INDEX

    def test_the_supervised_span_sits_at_the_end(self) -> None:
        from lexi_research.train.collate import IGNORE_INDEX

        example = build_corrector_example(_stub_tokenizer(), ROW)
        supervised = [i for i, label in enumerate(example.labels) if label != IGNORE_INDEX]
        assert supervised == list(range(supervised[0], len(example.labels)))

    def test_labels_match_input_ids_where_supervised(self) -> None:
        from lexi_research.train.collate import IGNORE_INDEX

        example = build_corrector_example(_stub_tokenizer(), ROW)
        for token, label in zip(example.input_ids, example.labels, strict=True):
            assert label in (IGNORE_INDEX, token)

    def test_it_is_shorter_than_the_stage_b_sequence(self) -> None:
        from lexi_research.train.collate import build_example

        tokenizer = _stub_tokenizer()
        stage_b_row = {
            **ROW,
            "target": "bright",
            "definition": "full of light",
            "pos": "adjective",
            "meaning": 3,
            "feedback": "Check the verb.",
        }
        mine = build_corrector_example(tokenizer, ROW)
        theirs = build_example(tokenizer, stage_b_row, thinking="off")
        assert len(mine.input_ids) < len(theirs.input_ids)

    def test_an_over_long_example_raises_rather_than_truncating(self) -> None:
        from lexi_research.train.collate import SequenceTooLong

        with pytest.raises(SequenceTooLong):
            build_corrector_example(_stub_tokenizer(), ROW, max_seq_len=8)

    def test_an_unknown_thinking_mode_raises(self) -> None:
        from lexi_research.train.collate import CollationError

        with pytest.raises(CollationError):
            build_corrector_example(_stub_tokenizer(), ROW, thinking="sideways")

    def test_the_prompt_half_is_the_corrector_prompt(self) -> None:
        messages = corrector_messages(ROW)
        assert messages == render_corrector_prompt(ROW["text"])
