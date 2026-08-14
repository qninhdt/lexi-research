"""LoRA target resolution across architectures.

The defect being fixed is a hardcoded `q/k/v/o_proj` list. Its failure mode is
silence: on a stack that names its projections anything else, the adapter
attaches to a fraction of the model — or to nothing — and the run still trains,
still logs a falling loss, and still saves a checkpoint.

So these tests run the resolver over several synthetic module trees rather than
a real checkpoint. The question is whether role classification and coverage
counting hold for each shape, and that must be answerable in CI without a GPU, a
download, or torch.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from lexi_research.train.modules import (
    ATTENTION,
    FEEDFORWARD,
    OTHER,
    PRESETS,
    Layout,
    TargetResolutionError,
    is_linear_like,
    resolve_target_modules,
)


class FakeLinear:
    """Duck-typed as a projection: `in_features` / `out_features`, like `nn.Linear`."""

    def __init__(self) -> None:
        self.in_features = 8
        self.out_features = 8


class FakeConv:
    """A depthwise convolution — no low-rank factorisation to insert."""

    def __init__(self) -> None:
        self.in_channels = 8
        self.out_channels = 8
        self.kernel_size = 4


class FakeEmbedding:
    def __init__(self) -> None:
        self.num_embeddings = 128
        self.embedding_dim = 8


class FakeModel:
    """The whole contract: `named_modules()` yielding `(path, module)`."""

    def __init__(self, modules: dict[str, Any]) -> None:
        self._modules = modules

    def named_modules(self) -> Iterator[tuple[str, Any]]:
        yield "", self
        yield from self._modules.items()


def _stack(
    layers: int,
    leaves: Any,
    *,
    prefix: str = "model.layers",
) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for index in range(layers):
        for leaf, factory in leaves(index).items():
            modules[f"{prefix}.{index}.{leaf}"] = factory()
    return modules


DENSE_LEAVES = {
    "self_attn.q_proj": FakeLinear,
    "self_attn.k_proj": FakeLinear,
    "self_attn.v_proj": FakeLinear,
    "self_attn.o_proj": FakeLinear,
    "mlp.gate_proj": FakeLinear,
    "mlp.up_proj": FakeLinear,
    "mlp.down_proj": FakeLinear,
}

#: A Gated-DeltaNet style layer: projections under a differently named attention
#: container, plus a depthwise conv that must not be targeted.
LINEAR_ATTN_LEAVES = {
    "linear_attn.in_proj_a": FakeLinear,
    "linear_attn.in_proj_b": FakeLinear,
    "linear_attn.in_proj_qkv": FakeLinear,
    "linear_attn.in_proj_z": FakeLinear,
    "linear_attn.out_proj": FakeLinear,
    "linear_attn.conv1d": FakeConv,
    "mlp.gate_proj": FakeLinear,
    "mlp.up_proj": FakeLinear,
    "mlp.down_proj": FakeLinear,
}

LAYERS = 32
FULL_ATTENTION_INTERVAL = 4

#: The legacy hardcoded list, kept as a fixture so the defect it caused stays
#: measurable rather than becoming folklore.
LEGACY_PATTERNS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def dense_model(layers: int = LAYERS) -> FakeModel:
    """Llama / Qwen2 / Mistral shape: one attention container, one MLP."""
    modules = _stack(layers, lambda _: DENSE_LEAVES)
    modules["model.embed_tokens"] = FakeEmbedding()
    modules["lm_head"] = FakeLinear()
    return FakeModel(modules)


def hybrid_model(layers: int = LAYERS) -> FakeModel:
    """Interleaved linear-attention and full-attention layers."""
    modules = _stack(
        layers,
        lambda index: (
            DENSE_LEAVES if (index + 1) % FULL_ATTENTION_INTERVAL == 0 else LINEAR_ATTN_LEAVES
        ),
    )
    modules["model.embed_tokens"] = FakeEmbedding()
    modules["lm_head"] = FakeLinear()
    return FakeModel(modules)


def moe_model(layers: int = 4, experts: int = 3) -> FakeModel:
    def leaves(_: int) -> dict[str, Any]:
        entries: dict[str, Any] = {
            "self_attn.q_proj": FakeLinear,
            "self_attn.o_proj": FakeLinear,
        }
        for expert in range(experts):
            for leaf in ("w1", "w2", "w3"):
                entries[f"block_sparse_moe.experts.{expert}.{leaf}"] = FakeLinear
        return entries

    return FakeModel(_stack(layers, leaves))


def vision_language_model(layers: int = 4, vision_layers: int = 2) -> FakeModel:
    modules = _stack(layers, lambda _: DENSE_LEAVES)
    modules |= _stack(
        vision_layers,
        lambda _: {"attn.qkv": FakeLinear, "attn.proj": FakeLinear, "mlp.fc1": FakeLinear},
        prefix="model.visual.blocks",
    )
    return FakeModel(modules)


def test_preset_names_are_the_documented_three() -> None:
    assert set(PRESETS) == {"attn", "attn+mlp", "all-linear"}


@pytest.mark.parametrize(
    "model_factory",
    [dense_model, hybrid_model, moe_model, vision_language_model],
    ids=["dense", "hybrid", "moe", "vision-language"],
)
def test_every_preset_resolves_on_every_architecture(model_factory) -> None:
    """No preset may depend on a module name a particular family happens to use."""
    model = model_factory()
    for preset in PRESETS:
        resolution = resolve_target_modules(model, preset)
        assert resolution.names
        assert len(resolution.layers(ATTENTION)) == resolution.total_layers


def test_attention_coverage_does_not_depend_on_the_container_name() -> None:
    """`linear_attn` is attention; the old list reached a quarter of these layers."""
    resolution = resolve_target_modules(hybrid_model(), "attn")
    assert resolution.total_layers == LAYERS
    assert len(resolution.layers(ATTENTION)) == LAYERS


def test_the_legacy_pattern_list_reaches_a_quarter_of_the_attention() -> None:
    """The defect, measured: hardcoded names miss every non-conforming layer."""
    resolution = resolve_target_modules(hybrid_model(), LEGACY_PATTERNS)
    assert len(resolution.layers(ATTENTION)) == LAYERS // FULL_ATTENTION_INTERVAL
    assert len(resolution.layers()) == LAYERS  # the MLPs are reached everywhere


def test_all_linear_covers_every_layer_of_a_hybrid_stack() -> None:
    resolution = resolve_target_modules(hybrid_model(), "all-linear")
    assert set(resolution.layers()) == set(range(LAYERS))
    assert len(resolution.layers(ATTENTION)) == LAYERS


def test_attn_only_excludes_the_feed_forward() -> None:
    resolution = resolve_target_modules(dense_model(2), "attn")
    assert all(target.role == ATTENTION for target in resolution.targets)
    assert not [name for name in resolution.names if ".mlp." in name]


def test_moe_experts_are_classified_as_feed_forward() -> None:
    resolution = resolve_target_modules(moe_model(), "attn+mlp")
    roles = {target.role for target in resolution.targets if "experts" in target.name}
    assert roles == {FEEDFORWARD}


def test_a_moe_router_is_never_adapted_by_a_preset() -> None:
    """Adapting the router perturbs expert assignment during training."""
    model = FakeModel(
        {
            "model.layers.0.mlp.gate": FakeLinear(),
            "model.layers.0.mlp.shared_expert_gate": FakeLinear(),
            "model.layers.0.mlp.experts.0.gate_proj": FakeLinear(),
            "model.layers.0.self_attn.q_proj": FakeLinear(),
        }
    )
    for preset in PRESETS:
        names = resolve_target_modules(model, preset).names
        assert "model.layers.0.mlp.gate" not in names
        assert "model.layers.0.mlp.shared_expert_gate" not in names

    everything = resolve_target_modules(model, "all-linear")
    assert "model.layers.0.mlp.experts.0.gate_proj" in everything.names
    assert [t.role for t in everything.targets if t.name.endswith(".gate")] == []


def test_the_router_is_reachable_by_an_explicit_pattern() -> None:
    """Excluded from the presets, not from the tool — A6 can still ablate it."""
    model = FakeModel({"model.layers.0.mlp.gate": FakeLinear()})
    assert resolve_target_modules(model, ["mlp.gate"]).names == ("model.layers.0.mlp.gate",)


def test_a_comma_separated_string_is_a_pattern_list() -> None:
    """So `--override train.target_modules=q_proj,o_proj` works without a file edit."""
    resolution = resolve_target_modules(dense_model(2), "self_attn.q_proj, self_attn.o_proj")
    assert {name.rsplit(".", 1)[-1] for name in resolution.names} == {"q_proj", "o_proj"}


def test_a_vision_tower_is_never_adapted() -> None:
    """It has its own blocks and its own attention, and this is a text task."""
    resolution = resolve_target_modules(vision_language_model(), "all-linear")
    assert not [name for name in resolution.names if "visual" in name]
    assert resolution.total_layers == 4


def test_a_depthwise_conv_is_never_targeted() -> None:
    resolution = resolve_target_modules(hybrid_model(), "all-linear")
    assert not [name for name in resolution.names if name.endswith("conv1d")]


def test_embeddings_and_the_head_are_outside_the_decoder_stack() -> None:
    resolution = resolve_target_modules(dense_model(2), "all-linear")
    assert "lm_head" not in resolution.names
    assert not [name for name in resolution.names if "embed" in name]


def test_is_linear_like_reads_the_projection_convention() -> None:
    assert is_linear_like(FakeLinear())
    assert not is_linear_like(FakeConv())
    assert not is_linear_like(FakeEmbedding())


def test_unknown_pattern_raises() -> None:
    """A no-op adapter trains to completion and reports a loss. Not possible."""
    with pytest.raises(TargetResolutionError, match="fictional_proj"):
        resolve_target_modules(dense_model(2), ["fictional_proj"])


def test_unknown_preset_raises() -> None:
    """A non-preset string is read as patterns, so a typo fails on the patterns."""
    with pytest.raises(TargetResolutionError, match="all-the-things"):
        resolve_target_modules(dense_model(2), "all-the-things")


def test_a_preset_whose_role_is_absent_raises() -> None:
    """A stack with no recognisable attention must fail, not adapt nothing."""
    model = FakeModel({"model.layers.0.mlp.up_proj": FakeLinear()})
    with pytest.raises(TargetResolutionError, match="matched none"):
        resolve_target_modules(model, "attn")
    assert resolve_target_modules(model, "attn+mlp").names == ("model.layers.0.mlp.up_proj",)


def test_an_empty_pattern_list_raises() -> None:
    with pytest.raises(TargetResolutionError, match="empty"):
        resolve_target_modules(dense_model(1), [])


def test_an_unreadable_stack_raises_rather_than_resolving_empty() -> None:
    model = FakeModel({"encoder.stage0.projection": FakeLinear()})
    with pytest.raises(TargetResolutionError, match="matched none"):
        resolve_target_modules(model, "all-linear")


def test_layout_is_configuration_not_code() -> None:
    """A stack with unconventional names is reachable without touching the source."""
    model = FakeModel(
        {
            "transformer.stack.0.mixer.wq": FakeLinear(),
            "transformer.stack.0.swiglu.w_in": FakeLinear(),
        }
    )
    layout = Layout(
        layer_containers=("stack",),
        attention_markers=("mixer",),
        feedforward_markers=("swiglu",),
    )
    resolution = resolve_target_modules(model, "attn+mlp", layout)
    assert len(resolution.targets) == 2
    assert resolution.total_layers == 1


def test_unrecognised_projections_are_kept_only_by_all_linear() -> None:
    model = FakeModel(
        {
            "model.layers.0.self_attn.q_proj": FakeLinear(),
            "model.layers.0.residual_gate": FakeLinear(),
        }
    )
    assert resolve_target_modules(model, "attn").names == ("model.layers.0.self_attn.q_proj",)
    everything = resolve_target_modules(model, "all-linear")
    assert "model.layers.0.residual_gate" in everything.names
    assert [t.role for t in everything.targets if t.name.endswith("residual_gate")] == [OTHER]


def test_layout_from_config_rejects_an_unknown_key() -> None:
    with pytest.raises(TargetResolutionError, match="attention_marker"):
        Layout.from_config({"attention_marker": ["attn"]})


def test_layout_from_config_defaults_when_absent() -> None:
    assert Layout.from_config(None) == Layout()


def test_names_are_full_module_paths_and_deduplicated() -> None:
    resolution = resolve_target_modules(hybrid_model(), LEGACY_PATTERNS)
    assert all(name.startswith("model.layers.") for name in resolution.names)
    assert len(resolution.names) == len(set(resolution.names))


def test_a_pattern_matches_on_a_dot_boundary_only() -> None:
    model = FakeModel(
        {
            "model.layers.0.self_attn.no_proj": FakeLinear(),
            "model.layers.0.self_attn.o_proj": FakeLinear(),
        }
    )
    resolution = resolve_target_modules(model, ["self_attn.o_proj"])
    assert resolution.names == ("model.layers.0.self_attn.o_proj",)


def test_resolution_is_deterministic() -> None:
    first = resolve_target_modules(hybrid_model(), "all-linear")
    second = resolve_target_modules(hybrid_model(), "all-linear")
    assert first.names == second.names


def test_summary_reports_the_number_that_matters() -> None:
    """The line a human reads to know whether the adapter actually attached."""
    summary = resolve_target_modules(hybrid_model(), "all-linear").summary()
    assert "attention 32/32" in summary
