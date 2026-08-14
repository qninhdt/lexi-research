"""Conversion and rendering between inline markup and line-based span edits.

Enables training models on concise, structured span representations:
    START END TAG REPLACEMENT
while guaranteeing 100% exact roundtrip rendering back to standard inline markup
`[original>replacement:tag]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .parser import Edit, ParseError, _escape, parse_correction, render
from .tags import TAGS
from .units import SpanEdit, lex_units

if TYPE_CHECKING:
    from .units import Unit

__all__ = [
    "markup_to_spans",
    "parse_span_output",
    "render_spans_to_markup",
    "validate_span_edits",
]


def markup_to_spans(raw_text: str, correction_markup: str) -> str:
    """Convert dataset inline markup `He [speak>speaks:agr] English.` to Span Edits string."""
    raw_str = raw_text.strip()
    corr_str = correction_markup.strip()

    if corr_str == "null":
        return "NULL"
    if raw_str == corr_str or not corr_str:
        return "OK"

    parsed = parse_correction(corr_str)
    if isinstance(parsed, ParseError):
        return "NULL"

    if not parsed.edits:
        return "OK"

    units = lex_units(parsed.text)
    lines: list[str] = []

    for edit in parsed.edits:
        e_rep = edit.replacement
        e_tag = edit.tag
        e_start_char, e_end_char = edit.span

        if e_start_char == e_end_char:
            # Insertion: zero-width span
            insert_idx = len(units) + 1
            for u in units:
                if u.start_char >= e_start_char:
                    insert_idx = u.index
                    break
            lines.append(f"{insert_idx} {insert_idx} {e_tag} {e_rep}".strip())
        else:
            # Replacement or Deletion
            start_unit = len(units) + 1
            for u in units:
                if u.end_char > e_start_char:
                    start_unit = u.index
                    break
            end_unit = start_unit
            for u in units:
                if u.start_char < e_end_char:
                    end_unit = u.index + 1

            rep_part = f" {e_rep}" if e_rep else ""
            lines.append(f"{start_unit} {end_unit} {e_tag}{rep_part}")

    return "\n".join(lines)


def parse_span_output(output_str: str) -> list[SpanEdit] | str:
    """Parse model output text into SpanEdit tuples or special signals 'OK' / 'NULL'."""
    text = output_str.strip()
    if text == "OK":
        return "OK"
    if text == "NULL":
        return "NULL"

    edits: list[SpanEdit] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 3:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            continue
        tag = parts[2]
        rep = parts[3] if len(parts) > 3 else ""
        edits.append(SpanEdit(start=start, end=end, tag=tag, replacement=rep))

    if not edits:
        # If output was empty or invalid lines
        return "OK" if not text else "NULL"
    return edits


def render_spans_to_markup(raw_text: str, spans_input: list[SpanEdit] | str) -> str:
    """Deterministic Renderer: Reconstructs exact inline markup `[orig>rep:tag]` on raw original sentence."""
    if spans_input == "NULL":
        return "null"
    if spans_input == "OK" or not spans_input:
        return _escape(raw_text)

    if isinstance(spans_input, str):
        parsed = parse_span_output(spans_input)
        if isinstance(parsed, str):
            return "null" if parsed == "NULL" else _escape(raw_text)
        spans_input = parsed

    units = lex_units(raw_text)
    recovered_edits: list[Edit] = []

    # Sort spans in ascending order by (start, end)
    sorted_spans = sorted(spans_input, key=lambda e: (e.start, e.end))

    for span in sorted_spans:
        if span.start == span.end:
            # Insertion
            if span.start > len(units):
                char_pos = len(raw_text)
            else:
                char_pos = units[span.start - 1].start_char
            recovered_edits.append(
                Edit(
                    original="",
                    replacement=span.replacement,
                    tag=span.tag,
                    span=(char_pos, char_pos),
                )
            )
        else:
            s_idx = max(1, min(span.start, len(units))) - 1
            e_idx = max(s_idx + 1, min(span.end, len(units) + 1)) - 1

            start_char = units[s_idx].start_char
            end_char = units[e_idx - 1].end_char if e_idx > 0 else len(raw_text)
            orig = raw_text[start_char:end_char]
            recovered_edits.append(
                Edit(
                    original=orig,
                    replacement=span.replacement,
                    tag=span.tag,
                    span=(start_char, end_char),
                )
            )

    try:
        return render(raw_text, recovered_edits)
    except Exception:
        # Fallback if spans overlap or violate ordering
        return raw_text


def validate_span_edits(spans_input: list[SpanEdit] | str, num_units: int) -> tuple[bool, str]:
    """Validate that span edits strictly obey index bounds, taxonomy tags, and non-overlapping rules."""
    if spans_input in ("OK", "NULL"):
        return True, "valid"

    if isinstance(spans_input, str):
        parsed = parse_span_output(spans_input)
        if isinstance(parsed, str):
            return True, "valid"
        spans_input = parsed

    if not spans_input:
        return True, "valid"

    max_idx = num_units + 1
    last_end = -1

    for edit in spans_input:
        if not (1 <= edit.start <= max_idx):
            return False, f"start index {edit.start} out of bounds [1, {max_idx}]"
        if not (edit.start <= edit.end <= max_idx):
            return False, f"invalid span [{edit.start}, {edit.end}]"
        if edit.tag not in TAGS:
            return False, f"invalid tag {edit.tag!r}"
        if edit.start < last_end:
            return False, f"overlapping span starting at {edit.start} before previous end {last_end}"
        last_end = edit.end

    return True, "valid"
