"""Locate the four answer spans in a training sequence.

Everything in this package depends on knowing exactly which tokens are the
reasoning, which are the correction, which are the meaning band, and which are
the feedback. A boundary that is off by one token silently trains the wrong
thing, and no reward curve looks unusual when it happens.

So the spans are **constructed, not discovered**. The sequence is built by
concatenating separately tokenised chunks, and each span is recorded as it is
appended. Locating spans afterwards — by decoding and string-matching, or by
assuming a prefix property across tokenisations — is how off-by-one boundaries
get in.

`json.dumps(..., sort_keys=True)` orders the fields `correction`, `feedback`,
`meaning`; the chunking below follows that order rather than the design's reading
order, because the bytes are what the model sees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lexi_research.train.collate import EMPTY_REASONING, THINKING_MODES, ChatTokenizer

#: Field order in the serialised answer. Fixed by `sort_keys=True`.
FIELD_ORDER = ("correction", "feedback", "meaning")


class SegmentError(ValueError):
    """The sequence could not be built with unambiguous spans."""


@dataclass(frozen=True)
class Span:
    """A half-open `[start, end)` token range."""

    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    @property
    def empty(self) -> bool:
        return self.end <= self.start

    def indices(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True)
class Segments:
    """Where each part of the sequence lives, in token positions."""

    input_ids: tuple[int, ...]
    prompt: Span
    reasoning: Span
    #: The serialised answer end to end, punctuation and field names included.
    #: Recorded rather than derived: `sort_keys=True` puts `feedback` between
    #: `correction` and `meaning`, so "first field to last field" is not the same
    #: range as "start of the object to its closing brace".
    answer: Span
    correction: Span
    meaning: Span
    feedback: Span

    def mask(self, *spans: Span) -> list[int]:
        """A 0/1 vector over the sequence, 1 inside any of `spans`."""
        flags = [0] * len(self.input_ids)
        for span in spans:
            for index in span.indices():
                if 0 <= index < len(flags):
                    flags[index] = 1
        return flags


def _encode(tokenizer: ChatTokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _answer_chunks(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The serialised answer, split so every field value is its own chunk.

    Rebuilt from `json.dumps` of each value rather than assembled by hand, so the
    escaping matches byte for byte what the collator supervises.
    """
    payload = {key: row[key] for key in FIELD_ORDER}
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    chunks: list[tuple[str, str]] = []
    cursor = 0
    for index, field in enumerate(FIELD_ORDER):
        needle = f'"{field}":'
        position = rendered.find(needle, cursor)
        if position < 0:  # pragma: no cover - json.dumps produced these keys
            raise SegmentError(f"serialised answer has no {field!r} field")
        value = json.dumps(payload[field], ensure_ascii=False)
        value_at = rendered.find(value, position + len(needle))
        if value_at < 0:
            raise SegmentError(f"cannot locate the {field!r} value in the serialised answer")
        chunks.append(("", rendered[cursor:value_at]))
        chunks.append((field, value))
        cursor = value_at + len(value)
        del index
    chunks.append(("", rendered[cursor:]))
    return chunks


def build_segments(
    tokenizer: ChatTokenizer,
    row: Mapping[str, Any],
    prompt_ids: Sequence[int],
    *,
    thinking: str = "on",
    reasoning: str | None = None,
) -> Segments:
    """Build the full sequence and record every span as it is appended.

    `reasoning` is the sampled reasoning text for an RL rollout. Left unset, the
    reasoning span is empty for `on` and `off`, and holds the empty block for
    `forced-empty` — which is what makes `R(empty)` computable with the same code
    path as `R(z)`.
    """
    if thinking not in THINKING_MODES:
        raise SegmentError(f"thinking={thinking!r}; expected one of {list(THINKING_MODES)}")

    ids = list(prompt_ids)
    prompt = Span(0, len(ids))

    reasoning_text = reasoning
    if reasoning_text is None and thinking == "forced-empty":
        reasoning_text = EMPTY_REASONING
    reasoning_start = len(ids)
    if reasoning_text:
        if thinking == "off":
            raise SegmentError("thinking='off' cannot carry a reasoning block")
        ids += _encode(tokenizer, reasoning_text)
    reasoning_span = Span(reasoning_start, len(ids))

    spans: dict[str, Span] = {}
    answer_start = len(ids)
    for field, text in _answer_chunks(row):
        start = len(ids)
        ids += _encode(tokenizer, text)
        if field:
            spans[field] = Span(start, len(ids))
    answer = Span(answer_start, len(ids))

    missing = [field for field in FIELD_ORDER if field not in spans]
    if missing:  # pragma: no cover - `_answer_chunks` emits all three
        raise SegmentError(f"no span located for {missing}")
    for field, span in spans.items():
        if span.empty:
            raise SegmentError(
                f"the {field!r} span is empty; a tokenizer that drops it would make "
                "its reward and its gradient silently vanish"
            )

    return Segments(
        input_ids=tuple(ids),
        prompt=prompt,
        reasoning=reasoning_span,
        answer=answer,
        correction=spans["correction"],
        meaning=spans["meaning"],
        feedback=spans["feedback"],
    )


def reward_mask(segments: Segments, scope: str) -> list[int]:
    """Which tokens the reward is computed over. Ablation A3.

    `feedback` is excluded by default and that exclusion is the design's central
    claim: feedback is voice and register, unverifiable, and rewarding it teaches
    the model to chase teacher phrasing rather than grading quality. The
    `full_answer` scope exists to test that claim, not as a setting to prefer.
    """
    if scope == "correction_meaning":
        return segments.mask(segments.correction, segments.meaning)
    if scope == "full_answer":
        return segments.mask(segments.correction, segments.meaning, segments.feedback)
    raise SegmentError(f"rl.reward_scope={scope!r}; expected 'correction_meaning' or 'full_answer'")


def policy_gradient_mask(segments: Segments) -> list[int]:
    """Only the reasoning receives policy gradient, in every track.

    The answer is supervised by cross-entropy against the teacher's. Letting the
    policy gradient reach it would put two objectives on the same tokens, and the
    tracks would stop being comparable.
    """
    return segments.mask(segments.reasoning)


def supervised_mask(segments: Segments) -> list[int]:
    """Cross-entropy covers the whole answer, feedback included."""
    return segments.mask(segments.correction, segments.meaning, segments.feedback)


__all__ = [
    "FIELD_ORDER",
    "SegmentError",
    "Segments",
    "Span",
    "build_segments",
    "policy_gradient_mask",
    "reward_mask",
    "supervised_mask",
]
