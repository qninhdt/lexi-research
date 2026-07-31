"""The judge, which is mostly a set of defences against measuring the judge.

A model asked to compare two texts has a position bias — many prefer whichever
came second — so a single-order pairwise test reports that bias as a win rate.
Presenting both orders turns it into a contradiction that can be counted and
discarded instead.

No teacher endpoint here: `ask` is injected, so every bias these tests describe
can be constructed exactly.
"""

from __future__ import annotations

import pytest

from lexi_research.eval.judge import Verdict, build_prompt, judge_pairs


def _row(uid: str, student: str, teacher: str) -> dict:
    return {
        "req_uid": uid,
        "text": "She speak very well.",
        "gold": {"feedback": teacher},
        "prediction": {"feedback": student},
    }


ROWS = [_row(f"r-{index}", f"student {index}", f"teacher {index}") for index in range(6)]


def _always(letter: str):
    async def ask(_messages):
        return letter

    return ask


def _prefers(text: str):
    """A judge with real taste: picks whichever option holds `text`."""

    async def ask(messages):
        content = messages[1]["content"]
        first = content.split("Feedback A: ")[1].split("\n")[0]
        return "A" if text in first else "B"

    return ask


async def test_a_judge_that_always_says_a_is_discarded_entirely() -> None:
    """Position bias, not preference: it picks a different text in each order."""
    result = await judge_pairs(ROWS, _always("A"))
    assert result["judge_discard_rate"] == 1.0
    assert result["consistent"] == 0
    assert result["judge_win_rate"] == 0.0


async def test_a_consistent_preference_survives_both_orders() -> None:
    result = await judge_pairs(ROWS, _prefers("student"))
    assert result["judge_discard_rate"] == 0.0
    assert result["judge_win_rate"] == 1.0


async def test_a_consistent_preference_for_the_teacher_scores_zero() -> None:
    result = await judge_pairs(ROWS, _prefers("teacher"))
    assert result["judge_discard_rate"] == 0.0
    assert result["judge_win_rate"] == 0.0


async def test_an_unreadable_reply_is_discarded_not_guessed() -> None:
    async def ask(_messages):
        return "I would rather not say."

    result = await judge_pairs(ROWS, ask)
    assert result["judge_discard_rate"] == 1.0


async def test_the_sample_size_is_honoured_and_reported() -> None:
    result = await judge_pairs(ROWS, _prefers("student"), sample=2)
    assert result["n"] == 2


async def test_sampling_is_seeded() -> None:
    first = await judge_pairs(ROWS, _prefers("student"), sample=3, seed=7)
    second = await judge_pairs(ROWS, _prefers("student"), sample=3, seed=7)
    assert first == second


async def test_rows_without_feedback_on_both_sides_are_skipped() -> None:
    rows = [{"req_uid": "x", "text": "hi", "gold": {}, "prediction": {"feedback": "only one"}}]
    result = await judge_pairs(rows, _prefers("student"))
    assert result["n"] == 0
    assert result["reliability"] == "weak"


async def test_the_result_is_always_tagged_weak() -> None:
    result = await judge_pairs(ROWS, _prefers("student"))
    assert result["reliability"] == "weak"


def test_the_prompt_names_both_options_and_asks_for_one_character() -> None:
    messages = build_prompt("She speak.", "left text", "right text")
    assert messages[0]["role"] == "system"
    assert "A or B" in messages[0]["content"]
    assert "left text" in messages[1]["content"]
    assert "right text" in messages[1]["content"]


def test_the_prompt_tells_the_judge_to_ignore_length_and_confidence() -> None:
    """Otherwise it rewards the model that writes more, which is not the question."""
    system = build_prompt("x", "a", "b")[0]["content"].lower()
    assert "length" in system
    assert "confident" in system


@pytest.mark.parametrize(
    ("first", "second", "consistent"),
    [("student", "student", True), ("student", "teacher", False), ("student", None, False)],
)
def test_verdict_consistency(first, second, consistent) -> None:
    assert Verdict(req_uid="x", first=first, second=second).consistent is consistent
