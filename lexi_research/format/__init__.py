"""Format core: the correction parser, the band calculator, and the validator.

Pure functions, no I/O beyond reading `band_config.json`. Everything downstream
— the teacher's post-decode check, the training-data builder, the eval harness,
and the serving shim — goes through this one module, so the correction format
has exactly one implementation.
"""

from .bands import (
    MAX_BAND,
    MIN_BAND,
    BandConfig,
    Bands,
    count_words,
    default_config_path,
    derive_bands,
    penalty,
)
from .parser import (
    EDIT_RE,
    Edit,
    ParseError,
    ParseOk,
    ParseResult,
    parse_correction,
    render,
    strip_markup,
)
from .span_converter import (
    markup_to_spans,
    parse_span_output,
    render_spans_to_markup,
    validate_span_edits,
)
from .tags import CONFUSABLE_PAIRS, GROUP_OF, TAGS, Tag, TagGroup, group_of
from .units import (
    UNIT_RE,
    SpanEdit,
    Unit,
    format_numbered_input,
    lex_units,
)
from .validate import ValidationError, ValidationOk, ValidationResult, validate_output

__all__ = [
    "CONFUSABLE_PAIRS",
    "EDIT_RE",
    "GROUP_OF",
    "MAX_BAND",
    "MIN_BAND",
    "TAGS",
    "UNIT_RE",
    "BandConfig",
    "Bands",
    "Edit",
    "ParseError",
    "ParseOk",
    "ParseResult",
    "SpanEdit",
    "Tag",
    "TagGroup",
    "Unit",
    "ValidationError",
    "ValidationOk",
    "ValidationResult",
    "count_words",
    "default_config_path",
    "derive_bands",
    "format_numbered_input",
    "group_of",
    "lex_units",
    "markup_to_spans",
    "parse_correction",
    "parse_span_output",
    "penalty",
    "render",
    "render_spans_to_markup",
    "strip_markup",
    "validate_output",
    "validate_span_edits",
]
