"""Parse, strip, and render the inline `[A>B:tag]` correction format.

The grader re-emits the whole learner sentence with edits marked inline, so a
clean sentence costs zero extra tokens and the frontend can render the string
directly without matching spans back against the input.

Nothing here raises on model output: `parse_correction` returns a typed failure
instead, because the caller's policy differs by context (drop the row when
building the dataset, retry at inference).

Canonical spacing for inserts: an insert token stands alone between spaces, so
`went [>to the:art] store` strips back to `went store`. Stripping an insert
removes one adjacent space; rendering puts it back. Non-canonical spacing
(`went [>to:art]store`) parses fine but re-renders in canonical form.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .tags import TAGS

#: An escaped bracket: `\[` or `\]`. Both brackets are escapable, so an edit
#: field can carry a literal bracket without colliding with the markup.
_ESCAPED = r"\\[\[\]]"

#: The one regex, applied at a `[` that is not escaped. Pinned by the design doc.
#:
#: Each field admits escaped brackets but never raw ones, which is what lets a
#: literal `]` appear inside a field (`[\[hi\]>hi:punc]`) while keeping the
#: closing `]` of the markup unambiguous. `>` terminates the original field and
#: `:` the replacement, so neither may appear raw there either.
EDIT_RE = re.compile(rf"\[((?:{_ESCAPED}|[^\[\]>])*)>((?:{_ESCAPED}|[^\[\]:])*):([a-z]+)\]")


@dataclass(frozen=True)
class Edit:
    """One marked edit, positioned against the *stripped* learner text.

    `span` is a half-open `[start, end)` range into the stripped text: the text
    an edit replaces or deletes, or a zero-width point for an insert.
    """

    original: str
    replacement: str
    tag: str
    span: tuple[int, int]

    @property
    def is_insert(self) -> bool:
        return self.original == ""

    @property
    def is_delete(self) -> bool:
        return self.replacement == ""


@dataclass(frozen=True)
class ParseOk:
    """A fully parsed correction plus the learner text recovered from it."""

    edits: list[Edit]
    text: str


@dataclass(frozen=True)
class ParseError:
    """Why a correction string could not be parsed.

    `code` is stable and machine-readable (used for reject accounting);
    `detail` is for humans reading a rejects table.
    """

    code: str
    detail: str


ParseResult = ParseOk | ParseError


def _unescape(raw: str) -> str:
    r"""Turn `\[` and `\]` back into bare brackets.

    Raw brackets cannot reach here — `EDIT_RE` only admits escaped ones inside a
    field — so this needs no failure mode of its own.
    """
    return re.sub(r"\\([\[\]])", r"\1", raw)


def parse_correction(correction: str) -> ParseResult:
    """Parse a correction string into edits plus the original learner text."""
    edits: list[Edit] = []
    text: list[str] = []
    length = 0  # running length of the stripped text, == sum(len(p) for p in text)
    i = 0
    n = len(correction)

    while i < n:
        char = correction[i]

        if char == "\\" and i + 1 < n and correction[i + 1] in "[]":
            text.append(correction[i + 1])
            length += 1
            i += 2
            continue

        if char != "[":
            text.append(char)
            length += 1
            i += 1
            continue

        match = EDIT_RE.match(correction, i)
        if match is None:
            return ParseError("malformed_edit", f"unparseable edit at offset {i}")

        original = _unescape(match.group(1))
        replacement = _unescape(match.group(2))
        tag = match.group(3)
        if tag not in TAGS:
            return ParseError("unknown_tag", f"tag {tag!r} is not in the taxonomy")
        if original == "" and replacement == "":
            return ParseError("empty_edit", f"empty edit at offset {i}")

        i = match.end()
        start = length

        if original == "":
            # An insert token occupies its own slot between spaces, so removing
            # it must also remove one of those spaces or the stripped text gains
            # whitespace the learner never wrote. Prefer the trailing space;
            # fall back to the leading one at end-of-string.
            preceded_by_space = length == 0 or (text and text[-1].endswith(" "))
            if preceded_by_space and i < n and correction[i] == " ":
                i += 1
            elif i >= n and text and text[-1].endswith(" "):
                text[-1] = text[-1][:-1]
                length -= 1
                start = length
        else:
            text.append(original)
            length += len(original)

        edits.append(
            Edit(original=original, replacement=replacement, tag=tag, span=(start, length))
        )

    return ParseOk(edits=edits, text="".join(text))


def strip_markup(correction: str) -> str:
    """Recover the learner text from a correction string.

    Raises `ValueError` on malformed markup — call `parse_correction` for
    untrusted model output, which reports the failure instead of raising.
    """
    result = parse_correction(correction)
    if isinstance(result, ParseError):
        raise ValueError(f"{result.code}: {result.detail}")
    return result.text


def _escape(raw: str) -> str:
    r"""Escape both brackets, so `_unescape(_escape(s)) == s` for any `s`.

    Escaping only `[` would leave a literal `]` inside a field able to close the
    markup early, so a rendered string would not parse back.
    """
    return re.sub(r"([\[\]])", r"\\\1", raw)


def render(text: str, edits: Sequence[Edit]) -> str:
    """Rebuild a correction string from learner text plus edits.

    Inverse of `parse_correction` for canonically spaced input. Raises
    `ValueError` on spans that are out of order, overlapping, or inconsistent
    with `text` — those come from our own code, never from a model.
    """
    out: list[str] = []
    cursor = 0

    for edit in sorted(edits, key=lambda e: e.span):
        start, end = edit.span
        if start < cursor or end < start or end > len(text):
            raise ValueError(f"span {edit.span} is out of order or out of range")
        if text[start:end] != edit.original:
            raise ValueError(f"span {edit.span} does not hold {edit.original!r}")

        # Learner text is escaped on the way out too, not just the edit fields:
        # a bare `[` in the rendered string would read as the start of markup.
        out.append(_escape(text[cursor:start]))
        token = f"[{_escape(edit.original)}>{_escape(edit.replacement)}:{edit.tag}]"

        if edit.is_insert:
            # An insert token gets its own slot between spaces — the space that
            # parsing removes. At the end of the sentence there is no following
            # word to borrow a space from, so it takes the leading one instead.
            out.append(" " + token if start == len(text) else token + " ")
        else:
            out.append(token)

        cursor = end

    out.append(_escape(text[cursor:]))
    return "".join(out)
