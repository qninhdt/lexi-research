"""Validate a correction-only payload — the four checks that do not need a sense.

`format.validate_output` runs six checks, and two of them (`meaning` in 0..4,
`feedback` a single non-empty sentence) are about fields a general learner corpus
cannot supply. Passing stage-A rows through it would mean inventing a meaning
band and a feedback line for a sentence with no target word, which is fabricating
labels rather than validating them.

So the four checks that *are* about the correction live here, and
`validate_output` calls nothing in this module — the six-check path stays the one
thing model output at inference is graded by, unchanged.

Check 3 is again the one that cannot be dropped: a correction re-emits the whole
sentence, so a model can quietly reword text it did not mark and nothing
downstream would notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .parser import Edit, ParseError, parse_correction
from .tags import TAGS


@dataclass(frozen=True)
class CorrectionOk:
    """A correction that passed all four checks."""

    edits: list[Edit] | None
    text: str

    @property
    def unparseable(self) -> bool:
        """True when the grader judged the sentence beyond correction."""
        return self.edits is None


@dataclass(frozen=True)
class CorrectionError:
    """Which check failed, and why. `code` is stable for reject accounting."""

    code: str
    detail: str


CorrectionResult = CorrectionOk | CorrectionError


def validate_correction(correction: Any, text: str) -> CorrectionResult:
    """Check a correction against the learner text it claims to correct.

    `None` is legal and means the sentence was judged unrecoverable. Unlike the
    six-check path there is no band to floor here, so the caller decides what an
    unparseable sentence means for its own accounting.
    """
    if correction is None:
        return CorrectionOk(edits=None, text=text)
    if not isinstance(correction, str):
        return CorrectionError(
            "correction_type", f"correction is {type(correction).__name__}, not a string"
        )

    parsed = parse_correction(correction)
    if isinstance(parsed, ParseError):
        return CorrectionError(parsed.code, parsed.detail)

    unknown = sorted({edit.tag for edit in parsed.edits} - TAGS)
    if unknown:
        return CorrectionError("unknown_tag", f"tags outside the taxonomy: {unknown}")

    if any(edit.original == "" and edit.replacement == "" for edit in parsed.edits):
        return CorrectionError("empty_edit", "an edit has neither an original nor a replacement")

    if parsed.text != text:
        return CorrectionError(
            "strip_mismatch",
            "the correction with its markup removed is not the learner's sentence",
        )

    return CorrectionOk(edits=parsed.edits, text=parsed.text)


__all__ = ["CorrectionError", "CorrectionOk", "CorrectionResult", "validate_correction"]
