"""Validate a grader payload against the six post-decode checks.

The same six checks run in both places model output is consumed — building the
dataset (fail → drop the row) and inference (fail → retry) — so policy stays
with the caller and only the verdict lives here.

Check 3 is the one that cannot be dropped. Because a correction re-emits the
whole sentence, a model can quietly reword parts it did not mark, and nothing
downstream would notice; requiring the stripped correction to equal the input
exactly is what closes that hole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .bands import MAX_BAND, MIN_BAND, BandConfig, Bands, derive_bands
from .parser import Edit, ParseError, parse_correction

#: A sentence break: terminal punctuation, whitespace, then the start of a new
#: sentence. A trailing terminator is fine, and so is an internal abbreviation —
#: requiring a capital or a digit after the space is what separates
#: `Good sense. But fix the verb.` (two sentences) from
#: `Use e.g. 'bright light' instead.` (one). Feedback is a single generated
#: sentence, so a lowercase continuation after a full stop is not a case worth
#: chasing at the cost of rejecting every abbreviation.
_SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z0-9]")


@dataclass(frozen=True)
class ValidationOk:
    """A payload that passed all six checks, with the derived bands attached."""

    edits: list[Edit] | None
    meaning: int
    feedback: str
    bands: Bands

    @property
    def unparseable(self) -> bool:
        """True when the grader judged the sentence beyond correction."""
        return self.edits is None


@dataclass(frozen=True)
class ValidationError:
    """Which check failed, and why.

    `code` is stable so rejects can be counted by reason across runs.
    """

    code: str
    detail: str


ValidationResult = ValidationOk | ValidationError


def validate_output(
    payload: dict[str, Any],
    input_text: str,
    config: BandConfig,
) -> ValidationResult:
    """Run the six checks in order, stopping at the first failure."""
    for field in ("correction", "meaning", "feedback"):
        if field not in payload:
            return ValidationError("missing_field", f"payload has no {field!r}")

    correction = payload["correction"]
    edits: list[Edit] | None = None

    if correction is not None:
        if not isinstance(correction, str):
            return ValidationError("bad_type", "correction must be a string or null")

        # Checks 1, 2 and 5: the parser rejects malformed markup, tags outside
        # the taxonomy, and edits with neither an original nor a replacement.
        parsed = parse_correction(correction)
        if isinstance(parsed, ParseError):
            return ValidationError(parsed.code, parsed.detail)

        # Check 3: stripping the markup must reproduce the input exactly.
        if parsed.text != input_text:
            return ValidationError(
                "text_altered",
                f"stripped correction {parsed.text!r} != input {input_text!r}",
            )
        edits = parsed.edits

    # Check 4. bool is an int subclass, and `True` must not pass as band 1.
    meaning = payload["meaning"]
    if isinstance(meaning, bool) or not isinstance(meaning, int):
        return ValidationError("bad_type", "meaning must be an integer")
    if not MIN_BAND <= meaning <= MAX_BAND:
        return ValidationError("meaning_range", f"meaning {meaning} outside {MIN_BAND}..{MAX_BAND}")

    # Check 6.
    feedback = payload["feedback"]
    if not isinstance(feedback, str):
        return ValidationError("bad_type", "feedback must be a string")
    feedback = feedback.strip()
    if not feedback:
        return ValidationError("feedback_empty", "feedback is empty")
    if _SENTENCE_BREAK.search(feedback):
        return ValidationError("feedback_multi_sentence", "feedback holds more than one sentence")

    return ValidationOk(
        edits=edits,
        meaning=meaning,
        feedback=feedback,
        bands=derive_bands(edits, input_text, config),
    )


__all__ = ["ValidationError", "ValidationOk", "ValidationResult", "validate_output"]
