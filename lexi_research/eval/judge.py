"""Teacher-as-judge for `feedback`, which has no verifiable ground truth.

Every safeguard here exists because a naive pairwise judge measures the judge.

Order is randomised and *both* orders of every pair are presented. A model that
prefers whichever answer came second scores 50% on a shuffled test and 100% on an
unshuffled one; presenting both orders makes that visible as a contradiction
rather than as a win. Contradictory verdicts are discarded and the discard rate is
reported — a high rate means the judge could not tell the two apart, which is
itself the finding.

The result is still tagged weak everywhere it appears.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lexi_research.teacher.schemas import ChatMsg

SYSTEM = (
    "You are comparing two pieces of feedback written for an English learner "
    "about one sentence. Judge only which feedback would help the learner more: "
    "is it accurate about the sentence, specific, and kind. Ignore length, "
    "formatting, and which one sounds more confident. Answer with exactly one "
    "character: A or B."
)


@dataclass(frozen=True)
class Verdict:
    """One pair, judged in both orders."""

    req_uid: str
    first: str | None
    second: str | None

    @property
    def consistent(self) -> bool:
        """The judge picked the same *text* regardless of the position it held."""
        return self.first is not None and self.second is not None and self.first == self.second


def build_prompt(text: str, left: str, right: str) -> list[ChatMsg]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Sentence: {text}\n\nFeedback A: {left}\n\nFeedback B: {right}\n\n"
                "Which feedback helps more? Answer A or B."
            ),
        },
    ]


def _read_choice(reply: str, *, swapped: bool) -> str | None:
    """Map a reply to `student` or `teacher`, accounting for the presented order."""
    letter = next((char for char in reply.strip().upper() if char in "AB"), None)
    if letter is None:
        return None
    first_is_student = not swapped
    if letter == "A":
        return "student" if first_is_student else "teacher"
    return "teacher" if first_is_student else "student"


async def judge_pairs(
    rows: Sequence[Mapping[str, Any]],
    ask: Any,
    *,
    sample: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Pairwise win-rate over a sample, each pair judged in both orders.

    `ask` is an async callable taking messages and returning the reply text, so
    this is testable without a teacher endpoint and works with any client.
    """
    usable = [
        row
        for row in rows
        if isinstance(row.get("prediction"), Mapping)
        and isinstance(row["prediction"].get("feedback"), str)
        and isinstance(row["gold"].get("feedback"), str)
    ]
    if not usable:
        return {"judge_win_rate": 0.0, "judge_discard_rate": 0.0, "n": 0, "reliability": "weak"}

    rng = random.Random(seed)
    picked = rng.sample(usable, min(sample, len(usable)))

    async def one(row: Mapping[str, Any]) -> Verdict:
        student = str(row["prediction"]["feedback"])
        teacher = str(row["gold"]["feedback"])
        text = str(row["text"])
        forward, backward = await asyncio.gather(
            ask(build_prompt(text, student, teacher)),
            ask(build_prompt(text, teacher, student)),
        )
        return Verdict(
            req_uid=str(row.get("req_uid", "")),
            first=_read_choice(str(forward), swapped=False),
            second=_read_choice(str(backward), swapped=True),
        )

    verdicts = await asyncio.gather(*(one(row) for row in picked))
    consistent = [verdict for verdict in verdicts if verdict.consistent]
    wins = sum(1 for verdict in consistent if verdict.first == "student")
    return {
        "judge_win_rate": wins / len(consistent) if consistent else 0.0,
        # A high discard rate does not mean the judge failed — it means the two
        # texts were not distinguishable, which is a result about the student.
        "judge_discard_rate": 1 - len(consistent) / len(verdicts),
        "n": len(verdicts),
        "consistent": len(consistent),
        "reliability": "weak",
    }


__all__ = ["SYSTEM", "Verdict", "build_prompt", "judge_pairs"]
