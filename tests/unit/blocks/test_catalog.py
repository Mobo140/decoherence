"""blocks.catalog: the blocks and parameters a UI can offer."""
from __future__ import annotations

import importlib
import inspect
import json
import typing

import pytest

from src.blocks import catalog as catalog_lib
from src.blocks import channels, recipe, system, target
from src.blocks.recipe import Param, block


@pytest.fixture(scope="module")
def cat():
    return catalog_lib.catalog()


def _find(cat, fn):
    return next(b for s in cat["stages"].values() for b in s
                if b["fn"] == fn)


def _param(cat, fn, name):
    return next(p for p in _find(cat, fn)["params"] if p["name"] == name)


# --- Negative cases ---


def test_parameter_without_annotation_raises():
    def bad(system, x=1):
        return x

    with pytest.raises(ValueError, match=r"bad\(x\).*no annotation"):
        catalog_lib._params(bad)


def test_unsupported_annotation_raises():
    def bad(system, x: dict = None):
        return x

    with pytest.raises(ValueError, match=r"no form for the annotation"):
        catalog_lib._params(bad)


def test_literal_of_mixed_kinds_raises():
    def bad(system, x: typing.Literal[1, "a"] = 1):
        return x

    with pytest.raises(ValueError, match="Literal values"):
        catalog_lib._params(bad)


@pytest.mark.parametrize("call, message", [
    (lambda: system.two_qubit(1.0, "BAD"), "interaction"),
    (lambda: channels.jump(system.single_qubit(1.0), "BAD", 0.1),
     "BAD"),
])
def test_literal_annotation_keeps_the_runtime_error(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_criterion_annotation_keeps_the_runtime_error():
    from src.blocks.evolve import States
    import numpy as np
    states = States(np.array([0.0, 1.0]), np.stack([np.eye(2) / 2] * 2))
    with pytest.raises(ValueError, match="criterion must be one of"):
        target.coherence_crossing(
            states, system.single_qubit(1.0), 0.5, criterion="nope")


# --- Coverage ---


def test_every_public_block_function_appears(cat):
    listed = {b["fn"] for s in cat["stages"].values() for b in s}
    expected = set()
    for stage in catalog_lib.STAGES:
        module = importlib.import_module(f"src.blocks.{stage}")
        for name, f in vars(module).items():
            if (inspect.isfunction(f) and f.__module__ == module.__name__
                    and not name.startswith("_")):
                expected.add(f"{stage}.{name}")
    helpers = {
        "channels.to_mesolve", "observe.model_layout", "observe.combine",
        "observe.ar1_filter", "target.interpolate_crossing",
        "target.first_crossing", "target.relative_level",
        "target.sudden_death_time", "target.energy_basis_coherence"}
    assert listed == expected - helpers


def test_every_listed_block_is_accepted_by_block(cat):
    for stage in cat["stages"].values():
        for b in stage:
            assert block(b["fn"]).fn == b["fn"]


def test_catalog_is_json_safe(cat):
    assert json.loads(json.dumps(cat)) == cat


def test_stage_keys(cat):
    assert list(cat["stages"]) == list(catalog_lib.STAGES)
    assert cat["evolve_output"] == recipe.EVOLVERS


# --- Injected parameters ---


def test_injected_parameters_are_excluded(cat):
    banned = {"rng", "system", "channels", "states", "obs", "record",
              "rho0", "times"}
    for stage in cat["stages"].values():
        for b in stage:
            assert not banned & {p["name"] for p in b["params"]}, b["fn"]


def test_t_max_is_listed_and_marked(cat):
    for fn in ("channels.bernstein_profile", "channels.ou_modulation"):
        p = _param(cat, fn, "t_max")
        assert p["injected"] == "t_max" and not p["required"]
    assert _param(cat, "channels.jump", "gamma")["injected"] is None


def test_sampler_keyword_parameters_are_not_listed(cat):
    assert _find(cat, "samplers.uniform")["params"] == []


# --- Choices come from Literal ---


@pytest.mark.parametrize("fn, name, choices", [
    ("system.two_qubit", "interaction", ["XXZ", "TFIM"]),
    ("channels.jump", "op_type", ["sigma_minus", "sigma_z"]),
    ("channels.kondo_bath", "dim", [1, 2, 3]),
    ("initial.parity_sector", "sector", [1, -1]),
    ("evolve.sequential_record", "basis", ["z"]),
    ("observe.finite_shots", "settings", ["local", "tomography"]),
    ("target.coherence_crossing", "criterion",
     ["coherence", "purity", "concurrence"]),
])
def test_choices_come_from_literal(cat, fn, name, choices):
    assert _param(cat, fn, name)["choices"] == choices


def test_choices_agree_with_the_runtime_constants():
    assert list(typing.get_args(target.Criterion)) == list(target.CRITERIA)


def test_free_parameters_have_no_choices(cat):
    assert "choices" not in _param(cat, "channels.jump", "gamma")


@pytest.mark.parametrize("fn, name, kind", [
    ("channels.jump", "gamma", "float"),
    ("channels.kondo_bath", "dim", "int"),
    ("observe.finite_shots", "n_shots", "int"),
    ("channels.ou_modulation", "modulate_constant", "bool"),
    ("channels.bernstein_profile", "coeffs", "tuple"),
    ("observe.gaussian_noise", "names", "tuple"),
    ("system.two_qubit", "interaction", "str"),
])
def test_kinds(cat, fn, name, kind):
    assert _param(cat, fn, name)["kind"] == kind


def test_defaults_required_and_nullable(cat):
    spacing = _param(cat, "channels.kondo_bath", "spacing")
    assert spacing["default"] == 1.0 and not spacing["required"]
    assert _param(cat, "channels.jump", "gamma")["required"]
    s = _param(cat, "channels.ou_modulation", "s")
    assert s["nullable"] and s["required"]
    assert _param(cat, "observe.gaussian_noise", "names")["default"] is None


# --- Roles and needs ---


def test_channel_roles_follow_the_first_parameter(cat):
    roles = {b["name"]: b["role"] for b in cat["stages"]["channels"]}
    assert roles == {
        "jump": "producer", "kondo_bath": "producer",
        "bernstein_profile": "modifier", "ou_modulation": "modifier"}
    for b in cat["stages"]["channels"]:
        assert b["role"] == recipe.channel_role(block(b["fn"]))


def test_observe_roles_follow_the_first_parameter(cat):
    for b in cat["stages"]["observe"]:
        assert b["role"] == recipe.observe_role(block(b["fn"]))
    assert _find(cat, "observe.ar1_noise")["role"] == "modifier"
    assert _find(cat, "observe.purity")["role"] == "producer"


def test_roles_are_null_outside_channels_and_observe(cat):
    for stage in ("system", "initial", "evolve", "target", "samplers"):
        assert {b["role"] for b in cat["stages"][stage]} <= {None}


def test_needs_tells_states_from_record(cat):
    assert _find(cat, "observe.purity")["needs"] == "states"
    assert _find(cat, "observe.record_series")["needs"] == "record"
    assert _find(cat, "observe.ar1_noise")["needs"] is None
    assert {b["needs"] for b in cat["stages"]["target"]} == {"states"}


def test_docstring_first_line(cat):
    assert _find(cat, "channels.jump")["doc"] == (
        "Builds the jump channel of each qubit with a constant rate.")


def test_a_catalog_block_with_catalog_defaults_builds(cat):
    b = _find(cat, "channels.kondo_bath")
    values = {p["name"]: 1.0 for p in b["params"] if p["required"]}
    values.update(dim=2, rho_J=Param("J"))
    assert block(b["fn"], **values).param_names() == {"J"}
