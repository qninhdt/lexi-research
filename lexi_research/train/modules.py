"""Resolve LoRA target modules by inspecting the loaded model.

The trainer previously carried a hardcoded `q/k/v/o_proj` + MLP list. Such a list
is silently wrong on any architecture that names its projections differently —
on a hybrid linear-attention stack it adapts the attention of a quarter of the
layers and none of the rest, while still producing a run that trains, logs a
falling loss, and saves an adapter. Nothing about the outcome says the adapter
was mostly absent.

So nothing here knows any model. Targets are selected by **role**: walk
`named_modules()`, keep the ones that look like a linear projection, attribute
each to a decoder layer, classify it as attention or feed-forward from its path,
and report the coverage. Which path components mean what is configuration
(`train.layout` in `params.yaml`), so a new architecture is a config edit rather
than a code edit.

`named_modules()` plus `in_features` / `out_features` is the whole contract, which
keeps this importable and testable without torch.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

ATTENTION = "attention"
FEEDFORWARD = "feedforward"
ROUTER = "router"
OTHER = "other"

#: Presets, as sets of roles rather than module names. `all-linear` takes every
#: linear projection inside the decoder stack, including ones whose path matches
#: no known convention — that is the point of the `other` role.
#:
#: No preset includes `router`. An MoE router is a linear layer, but adapting it
#: perturbs expert assignment during training, so it is excluded from the default
#: the way the standard QLoRA-on-MoE recipes exclude it. An explicit pattern list
#: reaches it for anyone who wants to ablate that.
PRESETS: dict[str, tuple[str, ...]] = {
    "attn": (ATTENTION,),
    "attn+mlp": (ATTENTION, FEEDFORWARD),
    "all-linear": (ATTENTION, FEEDFORWARD, OTHER),
}


@dataclass(frozen=True)
class Layout:
    """Path-component conventions for reading a decoder stack.

    Substring matches on a single dotted component, which is what makes one set
    of conventions cover `self_attn`, `linear_attn`, `attn` and `attention`, or
    `mlp`, `block_sparse_moe` and `feed_forward`, without enumerating them.
    """

    layer_containers: tuple[str, ...] = ("layers", "h", "blocks")
    attention_markers: tuple[str, ...] = ("attn", "attention")
    feedforward_markers: tuple[str, ...] = ("mlp", "ffn", "feed_forward", "experts", "moe")
    #: MoE routers, matched by *exact* component name rather than substring:
    #: `gate` is a router, `gate_proj` is an ordinary feed-forward projection.
    router_markers: tuple[str, ...] = ("gate", "router", "shared_expert_gate")
    #: Sub-towers that are not the language model. A vision encoder has its own
    #: blocks and its own attention, and adapting it for a text task is waste at
    #: best.
    excluded_markers: tuple[str, ...] = (
        "visual",
        "vision_tower",
        "vision_model",
        "image_encoder",
        "audio_tower",
    )

    @classmethod
    def from_config(cls, values: Mapping[str, Any] | None) -> Layout:
        if not values:
            return cls()
        fields = {
            "layer_containers",
            "attention_markers",
            "feedforward_markers",
            "router_markers",
            "excluded_markers",
        }
        unknown = set(values) - fields
        if unknown:
            raise TargetResolutionError(f"unknown layout keys {sorted(unknown)}")
        return cls(**{key: tuple(str(item) for item in values[key]) for key in values})

    def layer_index_re(self) -> re.Pattern[str]:
        containers = "|".join(re.escape(name) for name in self.layer_containers)
        return re.compile(rf"(?:^|\.)(?:{containers})\.(\d+)(?:\.|$)")


class TargetResolutionError(ValueError):
    """A preset or pattern that selected nothing, or an unreadable layout."""


class HasNamedModules(Protocol):
    def named_modules(self) -> Iterable[tuple[str, Any]]: ...


def is_linear_like(module: Any) -> bool:
    """A projection LoRA can factorise, identified without importing torch.

    `in_features` / `out_features` is the convention every `nn.Linear` and every
    quantised drop-in replacement follows. A depthwise convolution — which a
    Gated-DeltaNet layer carries next to its projections — has `in_channels`
    instead and is excluded here rather than by name, as are embeddings and
    norms.
    """
    return isinstance(getattr(module, "in_features", None), int) and isinstance(
        getattr(module, "out_features", None), int
    )


def _has_marker(components: Sequence[str], markers: Sequence[str]) -> bool:
    return any(marker in component for component in components for marker in markers)


@dataclass(frozen=True)
class Target:
    """One matched module: where it is and what it does."""

    name: str
    layer: int
    role: str


@dataclass(frozen=True)
class TargetResolution:
    """What was matched, and how much of the decoder stack it covers."""

    targets: tuple[Target, ...]
    selector: str
    total_layers: int

    @property
    def names(self) -> tuple[str, ...]:
        """Full module paths. PEFT accepts these directly."""
        return tuple(target.name for target in self.targets)

    def layers(self, role: str | None = None) -> tuple[int, ...]:
        """Layer indices covered, optionally restricted to one role."""
        return tuple(sorted({t.layer for t in self.targets if role is None or t.role == role}))

    def summary(self) -> str:
        """The line printed before training starts, read to spot a no-op adapter."""
        attention = len(self.layers(ATTENTION))
        feedforward = len(self.layers(FEEDFORWARD))
        return (
            f"{self.selector}: {len(self.targets)} modules, "
            f"attention {attention}/{self.total_layers}, "
            f"feed-forward {feedforward}/{self.total_layers}, "
            f"any {len(self.layers())}/{self.total_layers} layers"
        )


def _scan(model: HasNamedModules, layout: Layout) -> tuple[list[Target], int]:
    """Every linear projection inside the decoder stack, with its layer and role."""
    layer_re = layout.layer_index_re()
    found: list[Target] = []
    layer_indices: set[int] = set()

    for name, module in model.named_modules():
        if not name:
            continue
        components = name.split(".")
        if _has_marker(components, layout.excluded_markers):
            continue
        match = layer_re.search(name)
        if match is None:
            continue
        layer = int(match.group(1))
        layer_indices.add(layer)
        if not is_linear_like(module):
            continue
        tail = name[match.end() :].split(".")
        if tail[-1] in layout.router_markers:
            role = ROUTER
        elif _has_marker(tail, layout.attention_markers):
            role = ATTENTION
        elif _has_marker(tail, layout.feedforward_markers):
            role = FEEDFORWARD
        else:
            role = OTHER
        found.append(Target(name=name, layer=layer, role=role))

    return found, len(layer_indices)


def _matches(name: str, pattern: str) -> bool:
    """Suffix match on a dot boundary, so `o_proj` never matches `no_proj`."""
    return name == pattern or name.endswith("." + pattern)


def resolve_target_modules(
    model: HasNamedModules,
    spec: str | Sequence[str],
    layout: Layout | None = None,
) -> TargetResolution:
    """Select LoRA targets from the model's own module tree.

    `spec` is a preset name — `attn`, `attn+mlp`, `all-linear` — or a list of
    module-name patterns for an architecture the layout conventions misread. A
    string that is not a preset is read as a comma-separated pattern list, so the
    escape hatch is reachable through `--override` and not only by editing
    `params.yaml`.

    Raises if the selection is empty, or if any pattern matches nothing: an
    adapter attached to zero modules trains, converges, and teaches the model
    nothing.
    """
    layout = layout or Layout()
    candidates, total_layers = _scan(model, layout)

    if isinstance(spec, str) and spec in PRESETS:
        targets = [target for target in candidates if target.role in PRESETS[spec]]
        selector = spec
    else:
        raw = spec.split(",") if isinstance(spec, str) else spec
        patterns = tuple(str(item).strip() for item in raw if str(item).strip())
        if not patterns:
            raise TargetResolutionError("target module pattern list is empty")
        targets = []
        unmatched: list[str] = []
        for pattern in patterns:
            hits = [target for target in candidates if _matches(target.name, pattern)]
            if not hits:
                unmatched.append(pattern)
            targets.extend(hits)
        if unmatched:
            raise TargetResolutionError(
                f"target patterns matched no linear module: {sorted(unmatched)}. "
                f"The stack holds {len(candidates)} linear modules across "
                f"{total_layers} layers. Expected either one of the presets "
                f"{sorted(PRESETS)} or patterns that name modules this checkpoint "
                "actually has — not a no-op adapter."
            )
        selector = ",".join(patterns)

    if not targets:
        raise TargetResolutionError(
            f"selector {selector!r} matched none of the {len(candidates)} linear "
            f"modules found across {total_layers} layers. Either the preset's role "
            "is absent from this architecture or `train.layout` misreads its module "
            "names."
        )

    unique = {target.name: target for target in targets}
    return TargetResolution(
        targets=tuple(unique[name] for name in sorted(unique)),
        selector=selector,
        total_layers=total_layers,
    )


__all__ = [
    "ATTENTION",
    "FEEDFORWARD",
    "OTHER",
    "PRESETS",
    "ROUTER",
    "HasNamedModules",
    "Layout",
    "Target",
    "TargetResolution",
    "TargetResolutionError",
    "is_linear_like",
    "resolve_target_modules",
]
