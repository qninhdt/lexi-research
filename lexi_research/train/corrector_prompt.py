"""The stage-A prompt: correct a learner sentence, no sense and no bands.

Deliberately *not* in `teacher/prompts/`. `prompt_hash` fingerprints every
`.jinja` file in that directory and the hash is part of the teacher cache key, so
adding a template there would invalidate every cached teacher response and make
stage B pay again for calls it has already made. Measured, not assumed: dropping a
file into that directory changes the hash.

The separation is also the honest one. `render_grader_prompt` is a contract shared
by generation, eval, and serving — parity with inference is what makes the project
distillation. This prompt has no such duty: nothing serves it. It exists so stage
A can teach the markup and the tag set without a `meaning` rubric and a `feedback`
rule that a general learner corpus cannot supply labels for.

It is shorter for a reason that is measured rather than aesthetic: the grader
system prompt is ~4.3k characters and dominates both the sequence budget and the
prefill cost of every step. Stage A has ~20k rows to get through on one L4, so the
rubric it does not need is GPU time it does not spend.

The untrusted-input boundary is kept identical to the grader's. Learner text is
learner text in both stages, and a guard that differed between them would be a
gap someone eventually walks through.
"""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lexi_research.teacher.registry import _UNTRUSTED_GUARD, _sanitize
from lexi_research.teacher.schemas import ChatMsg

#: Where the stage-A templates live — inside the training package, not the
#: teacher's, so `prompt_hash` does not see them.
CORRECTOR_PROMPTS_DIR = Path(__file__).parent / "prompts"

#: The two templates that make up the stage-A prompt.
CORRECTOR_TEMPLATES: tuple[str, ...] = ("corrector_system.jinja", "corrector_user.jinja")

#: The system template per `rubric` mode.
#:
#: `full` explains all 16 tags; `terse` names them and leaves the model to learn
#: what each means from ~19k examples. The choice is worth measuring rather than
#: assuming, because it is not a style question: the learner sentence averages 14
#: tokens and the answer 14, so the full rubric is 97% of every sequence and the
#: difference between the two is roughly 17 GPU-hours and 1 on an L4.
RUBRIC_MODES: dict[str, str] = {
    "full": "corrector_system.jinja",
    "terse": "corrector_system_terse.jinja",
}

_ENV = Environment(
    loader=FileSystemLoader(CORRECTOR_PROMPTS_DIR),
    undefined=StrictUndefined,
    autoescape=False,  # noqa: S701 - prompts are text for a model, not HTML
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def corrector_prompt_hash() -> str:
    """Fingerprint of the stage-A templates, for the run manifest.

    Separate from `prompt_hash` so a stage-A prompt edit invalidates stage-A
    artifacts and nothing else. Covers every template on disk, both rubric modes
    included, so switching mode is visible in the artifact's lineage.
    """
    digest = hashlib.sha256()
    for name in sorted(path.name for path in CORRECTOR_PROMPTS_DIR.glob("*.jinja")):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((CORRECTOR_PROMPTS_DIR / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def render_corrector_prompt(
    text: str, *, nonce: str | None = None, rubric: str = "full"
) -> list[ChatMsg]:
    """Build the stage-A prompt for one learner sentence.

    `nonce` is injectable only so tests can assert byte-identical output; leave it
    unset in production so each example carries a fresh delimiter, matching the
    distribution the grader prompt draws from.
    """
    if rubric not in RUBRIC_MODES:
        raise ValueError(f"rubric={rubric!r}; expected one of {sorted(RUBRIC_MODES)}")
    token = nonce or secrets.token_hex(8)
    system = _ENV.get_template(RUBRIC_MODES[rubric]).render()
    guard = _UNTRUSTED_GUARD.format(nonce=token)
    user = _ENV.get_template("corrector_user.jinja").render(
        text=_sanitize(text),
        nonce=token,
    )
    return [
        {"role": "system", "content": f"{system}\n\n{guard}"},
        {"role": "user", "content": user},
    ]


__all__ = [
    "CORRECTOR_PROMPTS_DIR",
    "CORRECTOR_TEMPLATES",
    "RUBRIC_MODES",
    "corrector_prompt_hash",
    "render_corrector_prompt",
]
