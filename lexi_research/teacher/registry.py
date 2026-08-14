"""Prompt templates as versioned, hashed artifacts — and one way to render them.

Two things live here, and both exist to stop the same failure:

`render_grader_prompt` is the **only** function that builds the grading prompt.
Call 2, the eval harness, and the serving shim all call it. Prompt parity is
therefore a property of the code rather than a convention someone has to
remember, and a test asserts that no other module reaches for the grader
templates.

`prompt_hash` fingerprints every template on disk. It goes into the run manifest
and into the cache key, so editing a prompt invalidates the generated data that
depended on it instead of silently mixing two prompt versions in one dataset.

The learner sentence is untrusted input. It is wrapped in a nonce-delimited block
with a boundary rule in the system prompt, mirroring `lexi_ai.llm.guarded_messages`
(read-only reference). The nonce is fresh per call, so callers must key the cache
on the stable request identity rather than on the rendered messages.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .schemas import ChatMsg, DiversifySpec, SenseRef

#: Where the `.jinja` templates live. Part of the package, so an installed wheel
#: renders the same prompts as a source checkout.
PROMPTS_DIR = Path(__file__).parent / "prompts"

#: The templates that make up the inference prompt. Named so the parity test can
#: assert nothing outside this module loads them.
GRADER_TEMPLATES: tuple[str, ...] = ("grader_system.jinja", "grader_user.jinja")

#: `StrictUndefined` turns a missing variable into an error instead of an empty
#: string. A prompt that silently loses its rubric would still return plausible
#: labels, which is the worst possible way for this to fail.
_ENV = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    undefined=StrictUndefined,
    autoescape=False,  # noqa: S701 - prompts are text for a model, not HTML
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)

def template_names() -> tuple[str, ...]:
    """Every template on disk, sorted — the set `prompt_hash` covers."""
    return tuple(sorted(path.name for path in PROMPTS_DIR.glob("*.jinja")))


def prompt_hash() -> str:
    """Fingerprint of all prompt templates.

    Stable across runs and processes: it hashes file *bytes* in name order, with
    no dependence on filesystem ordering, mtimes, or the rendering environment.
    """
    digest = hashlib.sha256()
    for name in template_names():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((PROMPTS_DIR / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _render(name: str, /, **context: object) -> str:
    return _ENV.get_template(name).render(**context)


def render_grader_prompt(
    target: str,
    sense: SenseRef,
    text: str,
    *,
    nonce: str | None = None,
) -> list[ChatMsg]:
    """Build the grading prompt: THE inference prompt, used by call 2 too.

    Every consumer — dataset generation, eval, serving — must come through here.
    """
    system = _render("grader_system.jinja")
    user = _render(
        "grader_user.jinja",
        target=target,
        definition=sense.definition,
        pos=sense.pos,
        text=text,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_diversify_prompt(
    target: str,
    sense: SenseRef,
    specs: Sequence[DiversifySpec],
    profiles: dict[str, str],
) -> list[ChatMsg]:
    """Build call 1's prompt: K specs for one sense, K learner sentences out.

    `profiles` maps a profile id to its trait description. Everything here is a
    diversity knob — call 2 reads the resulting text blind and decides what is
    actually true of it, so nothing in this prompt is a label.
    """
    rows = [
        {
            "spec_id": spec.spec_id,
            "traits": profiles[spec.profile_id],
            "meaning_req": spec.meaning_req,
            "error_spec": spec.error_spec,
            "error_bias": ", ".join(spec.error_bias) or "none in particular",
        }
        for spec in specs
    ]
    return [
        {"role": "system", "content": _render("diversify_system.jinja")},
        {
            "role": "user",
            "content": _render(
                "diversify_user.jinja",
                target=target,
                definition=sense.definition,
                pos=sense.pos,
                specs=rows,
            ),
        },
    ]


__all__ = [
    "GRADER_TEMPLATES",
    "PROMPTS_DIR",
    "prompt_hash",
    "render_diversify_prompt",
    "render_grader_prompt",
    "template_names",
]
