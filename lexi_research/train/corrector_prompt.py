"""The stage-A prompt: correct a learner sentence, no sense and no bands.

Deliberately *not* in `teacher/prompts/`. `prompt_hash` fingerprints every
`.jinja` file in that directory and the hash is part of the teacher cache key, so
adding a template there would invalidate every cached teacher response and make
stage B pay again for calls it has already made.

This prompt teaches the markup and tag set from a general learner corpus with a
concise, token-efficient template that omits verbose rubrics and untrusted guards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lexi_research.teacher.schemas import ChatMsg

#: Where the stage-A templates live.
CORRECTOR_PROMPTS_DIR = Path(__file__).parent / "prompts"

#: The templates that make up the stage-A prompt.
CORRECTOR_TEMPLATES: tuple[str, ...] = ("corrector_system.jinja", "corrector_user.jinja")

_ENV = Environment(
    loader=FileSystemLoader(CORRECTOR_PROMPTS_DIR),
    undefined=StrictUndefined,
    autoescape=False,  # noqa: S701 - prompts are text for a model, not HTML
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def corrector_prompt_hash() -> str:
    """Fingerprint of the stage-A templates, for the run manifest."""
    digest = hashlib.sha256()
    for name in sorted(path.name for path in CORRECTOR_PROMPTS_DIR.glob("*.jinja")):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((CORRECTOR_PROMPTS_DIR / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def render_corrector_prompt(text: str) -> list[ChatMsg]:
    """Build the concise stage-A prompt for one learner sentence."""
    from lexi_research.format.units import format_numbered_input

    system = _ENV.get_template("corrector_system.jinja").render()
    cleaned = text.strip()
    numbered = format_numbered_input(cleaned) if "\n" not in cleaned else cleaned
    user = _ENV.get_template("corrector_user.jinja").render(text=numbered)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = [
    "CORRECTOR_PROMPTS_DIR",
    "CORRECTOR_TEMPLATES",
    "corrector_prompt_hash",
    "render_corrector_prompt",
]
