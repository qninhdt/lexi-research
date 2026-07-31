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
from .tags import CONFUSABLE_PAIRS, GROUP_OF, TAGS, Tag, TagGroup, group_of
from .validate import ValidationError, ValidationOk, ValidationResult, validate_output

__all__ = [
    "CONFUSABLE_PAIRS",
    "EDIT_RE",
    "GROUP_OF",
    "MAX_BAND",
    "MIN_BAND",
    "TAGS",
    "BandConfig",
    "Bands",
    "Edit",
    "ParseError",
    "ParseOk",
    "ParseResult",
    "Tag",
    "TagGroup",
    "ValidationError",
    "ValidationOk",
    "ValidationResult",
    "count_words",
    "default_config_path",
    "derive_bands",
    "group_of",
    "parse_correction",
    "penalty",
    "render",
    "strip_markup",
    "validate_output",
]
