"""Build `smoke_50.jsonl`, the fixture the CPU acceptance gate trains on.

Rows are authored as correction markup and the learner text is *derived* from it
by `strip_markup`, so the two can never disagree — check 3 of the validator is
satisfied by construction rather than by proofreading. Every row is then run
through `validate_output`, and the set is checked for the coverage `lexi smoke`
asserts: all five meaning bands, both tag groups, a null correction, a multiword
target, and a clean sentence re-emitted verbatim.

The text is invented for this file. Nothing here derives from the private
dictionary source, which is what lets the fixture live in Git and run in CI.

    uv run python ops/fixtures/build_smoke_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from lexi_research.cli.smoke import check_fixture
from lexi_research.format.parser import strip_markup

# (target, definition, pos, correction, meaning, feedback)
# `correction` is None where the sentence is beyond correction; the learner text
# then follows in place of the markup.
ROWS: list[tuple[str, str, str, str | None, int, str]] = [
    (
        "speak",
        "to say words, to talk to someone",
        "verb",
        "She [speak>spoke:tense] very well in the meeting yesterday.",
        4,
        "Great use of the word, but the past tense is needed here.",
    ),
    (
        "speak",
        "to say words, to talk to someone",
        "verb",
        "I want to speak [>to:prep] my teacher about the homework.",
        3,
        "Good meaning, though 'speak' needs 'to' before the person.",
    ),
    (
        "speak",
        "to say words, to talk to someone",
        "verb",
        None,
        0,
        "The sentence does not use 'speak' with its dictionary meaning at all.",
    ),
    (
        "speak",
        "to say words, to talk to someone",
        "verb",
        "Can you speak louder please?",
        3,
        "Natural and correct, though this leans toward volume rather than conversation.",
    ),
    (
        "run",
        "to move quickly on foot",
        "verb",
        "He [run>ran:tense] very fast in the race last week.",
        4,
        "Perfect meaning usage; only the past tense form needs fixing.",
    ),
    (
        "run",
        "to move quickly on foot",
        "verb",
        "I like to run in [>the:art] park every morning.",
        4,
        "Excellent meaning, and a missing article is the only issue.",
    ),
    (
        "run",
        "to move quickly on foot",
        "verb",
        "She [runs>has run:tense] the company since 2010.",
        1,
        "This uses 'run' in the sense of managing rather than moving on foot.",
    ),
    (
        "run",
        "to move quickly on foot",
        "verb",
        "The children are running happily in the playground.",
        4,
        "Perfect sentence with the correct meaning and no errors.",
    ),
    (
        "bright",
        "full of light; shining strongly",
        "adjective",
        "The sun is very bright today so I need sunglasses.",
        4,
        "Perfectly natural sentence with the correct meaning.",
    ),
    (
        "bright",
        "full of light; shining strongly",
        "adjective",
        "She is a bright student who always [get>gets:agr] good grades.",
        1,
        "'Bright' here means intelligent rather than full of light.",
    ),
    (
        "bright",
        "full of light; shining strongly",
        "adjective",
        "The room was too bright because of the big windows.",
        4,
        "Correct meaning and natural grammar throughout.",
    ),
    (
        "bright",
        "full of light; shining strongly",
        "adjective",
        None,
        0,
        "'Bright' is used as a verb here, which this adjective sense does not allow.",
    ),
    (
        "break",
        "to separate into pieces as a result of force",
        "verb",
        "He [breaked>broke:tense] the window with a ball.",
        4,
        "Correct meaning; 'break' is irregular, so the past tense is 'broke'.",
    ),
    (
        "break",
        "to separate into pieces as a result of force",
        "verb",
        "Be careful, you will break the glass if you drop it.",
        4,
        "Perfect sentence with an accurate meaning.",
    ),
    (
        "break",
        "to separate into pieces as a result of force",
        "verb",
        "I need a break from studying.",
        0,
        "'Break' is a noun meaning a pause here, not the verb sense given.",
    ),
    (
        "break",
        "to separate into pieces as a result of force",
        "verb",
        "My phone screen [break>broke:tense] when I [drop>dropped:tense] it.",
        4,
        "Great use of the meaning, but both verbs need the past tense.",
    ),
    (
        "heavy",
        "weighing a lot; difficult to lift or move",
        "adjective",
        "This box is too heavy for me to carry alone.",
        4,
        "Perfectly correct in both meaning and grammar.",
    ),
    (
        "heavy",
        "weighing a lot; difficult to lift or move",
        "adjective",
        "There was heavy rain last night and the road is [flood>flooded:form].",
        2,
        "'Heavy rain' means intense rather than physically weighing a lot.",
    ),
    (
        "heavy",
        "weighing a lot; difficult to lift or move",
        "adjective",
        "The heavy books [make>made:tense] my bag very hard to carry.",
        4,
        "Excellent use of 'heavy' in the physical weight sense.",
    ),
    (
        "heavy",
        "weighing a lot; difficult to lift or move",
        "adjective",
        "I feel very heavy after [eat>eating:form] too much.",
        3,
        "The meaning is close, though the weight here is figurative.",
    ),
    (
        "catch",
        "to capture or seize something that is moving",
        "verb",
        "He [catched>caught:tense] the ball during the game.",
        4,
        "Correct meaning; 'catch' is irregular, so the past tense is 'caught'.",
    ),
    (
        "catch",
        "to capture or seize something that is moving",
        "verb",
        "I need to catch the bus before it [leave>leaves:agr].",
        2,
        "'Catch the bus' is idiomatic for boarding rather than seizing.",
    ),
    (
        "catch",
        "to capture or seize something that is moving",
        "verb",
        "The cat [>is:other] trying to catch the mouse in the kitchen.",
        4,
        "Great meaning usage, but the auxiliary verb is missing.",
    ),
    (
        "catch",
        "to capture or seize something that is moving",
        "verb",
        "She [catch>caught:tense] a cold last winter and was ill for weeks.",
        1,
        "'Catch a cold' is an idiom about illness rather than seizing something.",
    ),
    (
        "light",
        "the natural agent that makes things visible",
        "noun",
        "The light from the sun [make>makes:agr] the flowers beautiful.",
        4,
        "Excellent use of 'light' as the natural agent for visibility.",
    ),
    (
        "light",
        "the natural agent that makes things visible",
        "noun",
        "Please turn on the light, it is very dark here.",
        3,
        "Correct grammar, though 'light' here is an electric lamp.",
    ),
    (
        "light",
        "the natural agent that makes things visible",
        "noun",
        "This bag is very light, I can carry it easily.",
        0,
        "'Light' is an adjective meaning not heavy rather than the given noun.",
    ),
    (
        "light",
        "the natural agent that makes things visible",
        "noun",
        "The light in the morning [help>helps:agr] me wake up naturally.",
        4,
        "Perfect meaning, and only subject-verb agreement needs fixing.",
    ),
    (
        "sharp",
        "having an edge or point that can cut or pierce",
        "adjective",
        "Be careful with that sharp knife, you might cut yourself.",
        4,
        "Perfect sentence with an exact meaning match.",
    ),
    (
        "sharp",
        "having an edge or point that can cut or pierce",
        "adjective",
        "She has a sharp mind and [solve>solves:agr] problems quickly.",
        1,
        "'Sharp mind' means intelligent rather than able to cut.",
    ),
    (
        "sharp",
        "having an edge or point that can cut or pierce",
        "adjective",
        "The sharp edge of the paper cut my finger.",
        4,
        "Correct meaning and natural phrasing.",
    ),
    (
        "sharp",
        "having an edge or point that can cut or pierce",
        "adjective",
        None,
        0,
        "'Sharp' is used as a verb, and the correct word would be 'sharpen'.",
    ),
    (
        "go",
        "to move or travel to a place",
        "verb",
        "Where are you going[.>?:punc]",
        4,
        "The meaning is right, but a question needs a question mark.",
    ),
    (
        "receive",
        "to get something that someone gives you",
        "verb",
        "I [recieved>received:sp] a letter from my friend last week.",
        4,
        "Correct meaning, with only a spelling slip to fix.",
    ),
    (
        "book",
        "a set of printed pages held together in a cover",
        "noun",
        "There are three [book>books:num] on the small table.",
        4,
        "Right meaning; a plural noun is needed after 'three'.",
    ),
    (
        "brother",
        "a boy or man with the same parents as you",
        "noun",
        "This is my [brother>brother's:poss] new car.",
        4,
        "Correct meaning, but the possessive form is required here.",
    ),
    (
        "turn off",
        "to stop a machine or light working",
        "phrasal verb",
        "Please turn [of>off:part] the light when you leave the room.",
        4,
        "Exactly the right phrasal verb, with only the particle misspelt.",
    ),
    (
        "give",
        "to hand something to someone",
        "verb",
        "The teacher gave the book to [I>me:pron] yesterday morning.",
        4,
        "Correct meaning; the object form of the pronoun is needed.",
    ),
    (
        "late",
        "arriving after the expected time",
        "adjective",
        "She [always is>is always:order] late for the morning class.",
        4,
        "Right meaning, but the adverb belongs after the verb 'to be'.",
    ),
    (
        "decision",
        "a choice that you make after thinking",
        "noun",
        "We need to [do>make:coll] a decision before Friday afternoon.",
        3,
        "The meaning is right, though 'make' is the verb that collocates.",
    ),
    (
        "fun",
        "enjoyable and giving pleasure",
        "adjective",
        "The film was very [funny>fun:word] to watch with my friends.",
        2,
        "'Funny' means amusing, which is a different word from 'fun'.",
    ),
    (
        "have",
        "to feel or experience something",
        "verb",
        "[I am having>I have:unnat] a headache since this morning.",
        3,
        "The meaning is fine, but a stative use sounds unnatural in the continuous.",
    ),
    (
        "look after",
        "to take care of someone or something",
        "phrasal verb",
        "She looks after her little brother every afternoon.",
        4,
        "Perfectly natural sentence with the exact meaning.",
    ),
    (
        "heavy rain",
        "a large amount of rain falling",
        "collocation",
        "There was [strong>heavy:coll] rain during the whole night.",
        3,
        "The idea is right, but 'heavy' is the adjective that collocates with rain.",
    ),
    (
        "make up your mind",
        "to decide something",
        "idiom",
        "You should make up your mind before the shop [close>closes:agr].",
        4,
        "The idiom is used correctly, with only an agreement slip.",
    ),
    (
        "cold",
        "having a low temperature",
        "adjective",
        "The water in the lake was very cold this morning.",
        4,
        "Natural sentence with the correct meaning.",
    ),
    (
        "cold",
        "having a low temperature",
        "adjective",
        "He gave me a very cold [look>look:unnat] after the meeting.",
        1,
        "'Cold' here describes an unfriendly manner rather than temperature.",
    ),
    (
        "table",
        "a piece of furniture with a flat top and legs",
        "noun",
        "We put the plates on [>the:art] table before dinner.",
        4,
        "Correct meaning, and only the article is missing.",
    ),
    (
        "table",
        "a piece of furniture with a flat top and legs",
        "noun",
        "The teacher showed us a table of results in [class>the class:art].",
        1,
        "'Table' means an arrangement of data here rather than furniture.",
    ),
    (
        "open",
        "to move something so that it is no longer closed",
        "verb",
        "Could you open the window[>,:punc] please?",
        4,
        "Right meaning; a comma is needed before 'please' here.",
    ),
]


def build() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target, definition, pos, correction, meaning, feedback in ROWS:
        text = strip_markup(correction) if correction is not None else _plain_text(target)
        rows.append(
            {
                "target": target,
                "definition": definition,
                "pos": pos,
                "text": text,
                "correction": correction,
                "meaning": meaning,
                "feedback": feedback,
            }
        )
    return rows


#: Learner text for the rows judged beyond correction, where there is no markup
#: to derive it from.
UNPARSEABLE_TEXT = {
    "speak": "The dog speaks the bone every morning.",
    "bright": "I bright my shoes before going to school.",
    "sharp": "I sharp the pencil before the exam start.",
}


def _plain_text(target: str) -> str:
    return UNPARSEABLE_TEXT[target]


def main() -> int:
    rows = build()
    print(check_fixture(rows))
    destination = Path(__file__).with_name("smoke_50.jsonl")
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
