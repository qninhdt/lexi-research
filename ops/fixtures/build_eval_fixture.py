"""Build the eval fixtures: a predictions file and the ceiling it is scored against.

The harness is what Phase 4's null result will rest on, so it is established
against inputs whose answers are known before any model exists. The predictions
here are hand-written to exercise one case each — an exact hit, a right span with
a confusable tag, a right span with a cross-tier tag, a missed edit, a spurious
edit, an unparseable answer, and a clean sentence — so a metric that regresses
moves a number a test names.

    uv run python ops/fixtures/build_eval_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (req_uid, text, gold correction, predicted correction, gold meaning,
#  predicted meaning, confidence, note)
ROWS = [
    (
        "p-000",
        "She speak very well today.",
        "She [speak>spoke:tense] very well today.",
        "She [speak>spoke:tense] very well today.",
        4,
        4,
        0.95,
        "exact hit on span and tag",
    ),
    (
        "p-001",
        "He run fast every morning.",
        "He [run>runs:agr] fast every morning.",
        "He [run>runs:tense] fast every morning.",
        4,
        4,
        0.90,
        "right span, tag from the same weight tier",
    ),
    (
        "p-002",
        "I recieved a letter yesterday.",
        "I [recieved>received:sp] a letter yesterday.",
        "I [recieved>received:order] a letter yesterday.",
        4,
        3,
        0.55,
        "right span, tag from a different weight tier",
    ),
    (
        "p-003",
        "The room is very bright today.",
        "The room is very bright today.",
        "The room is very bright today.",
        4,
        4,
        0.99,
        "clean sentence, both sides agree",
    ),
    (
        "p-004",
        "There are three book on the table.",
        "There are three [book>books:num] on the table.",
        "There are three book on the table.",
        4,
        4,
        0.70,
        "missed edit",
    ),
    (
        "p-005",
        "This is my brother car.",
        "This is my [brother>brother's:poss] car.",
        "This is [my>the:art] [brother>brother's:poss] car.",
        4,
        2,
        0.40,
        "one hit, one spurious edit",
    ),
    (
        "p-006",
        "The dog speaks the bone every morning.",
        None,
        None,
        0,
        0,
        0.85,
        "both sides judged it beyond correction",
    ),
    (
        "p-007",
        "She has a sharp mind and solve problems.",
        "She has a sharp mind and [solve>solves:agr] problems.",
        "She has a sharp mind and [solve>solves:agr] problems.",
        1,
        2,
        0.30,
        "exact edit, meaning off by one",
    ),
]

GOLD_FEEDBACK = "The meaning is right, with one grammar point to fix."
PREDICTED_FEEDBACK = "Good meaning, and one grammar point needs fixing."


def build() -> list[dict[str, object]]:
    rows = []
    for uid, text, gold, predicted, gold_band, predicted_band, confidence, note in ROWS:
        rows.append(
            {
                "req_uid": uid,
                "text": text,
                "target": text.split()[1],
                "definition": "a fixture row",
                "pos": "verb",
                "gold": {
                    "correction": gold,
                    "meaning": gold_band,
                    "feedback": GOLD_FEEDBACK,
                },
                "prediction": {
                    "correction": predicted,
                    "meaning": predicted_band,
                    "feedback": PREDICTED_FEEDBACK,
                },
                "meaning_confidence": confidence,
                "retries": 0,
                "note": note,
            }
        )
    return rows


#: Teacher self-consistency, as `data/pilot_gate.py` measures it. Invented for the
#: fixture, and deliberately below 1.0: a ceiling of 1.0 would let a normalised
#: metric read the same as an unnormalised one and hide a bug in the
#: normalisation.
CEILING = {
    "meaning_qwk": 0.82,
    "correction_edit_f1": 0.74,
    "pairs": 200,
    "source": "fixture — invented, not measured",
}


def main() -> int:
    rows = build()
    predictions = HERE / "eval_predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ceiling = HERE / "eval_ceiling.json"
    ceiling.write_text(
        json.dumps(CEILING, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} predictions to {predictions} and the ceiling to {ceiling}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
