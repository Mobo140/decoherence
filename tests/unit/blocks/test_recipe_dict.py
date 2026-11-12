"""blocks.recipe: the JSON round trip of blocks and recipes."""
import copy
import json

import pytest

from src.blocks import catalog as catalog_lib
from src.blocks import recipe
from src.blocks.recipe import (
    Param, block, block_from_dict, block_to_dict, recipe_from_dict,
    recipe_to_dict)


def _good_dict():
    return {
        "system": {"fn": "system.single_qubit", "params": {"omega": 1.0}},
        "channels": [{"fn": "channels.jump", "params": {
            "op_type": "sigma_minus", "gamma": 0.2}}],
        "initial": {"fn": "initial.random_pure"},
        "observe": [{"fn": "observe.pauli_means"},
                    {"fn": "observe.purity"},
                    {"fn": "observe.coherence_l1"}],
        "target": {"fn": "target.coherence_crossing",
                   "params": {"threshold": 0.5}},
        "t_max": 5.0, "dt": 0.25,
    }


def _through_json(data):
    return json.loads(json.dumps(data))


# --- Negative cases: recipe_from_dict ---


@pytest.mark.parametrize("data", [None, [], "recipe", 3])
def test_recipe_must_be_an_object(data):
    with pytest.raises(ValueError, match="must be an object"):
        recipe_from_dict(data)


@pytest.mark.parametrize(
    "field", ["system", "channels", "initial", "observe", "target"])
def test_missing_required_field_is_named(field):
    d = _good_dict()
    del d[field]
    with pytest.raises(ValueError, match=f"^{field}: the field is missing"):
        recipe_from_dict(d)


def test_unknown_recipe_field_is_named():
    d = _good_dict()
    d["sampler_block"] = None
    with pytest.raises(ValueError, match="Unknown recipe fields.*sampler_b"):
        recipe_from_dict(d)


def test_unknown_block_field_names_the_stage():
    d = _good_dict()
    d["system"]["args"] = {}
    with pytest.raises(ValueError, match=r"^system: unknown fields.*args"):
        recipe_from_dict(d)


def test_block_without_fn_names_the_stage():
    d = _good_dict()
    del d["channels"][0]["fn"]
    with pytest.raises(ValueError, match=r"^channels\[0\]: .*'fn'"):
        recipe_from_dict(d)


def test_unknown_function_names_stage_and_index():
    d = _good_dict()
    d["observe"][1]["fn"] = "observe.nope"
    with pytest.raises(ValueError, match=r"^observe\[1\]: .*observe.nope"):
        recipe_from_dict(d)


def test_unknown_parameter_names_stage_and_parameter():
    d = _good_dict()
    d["channels"][0]["params"]["rate"] = 1.0
    with pytest.raises(ValueError, match=r"^channels\[0\]: .*'rate'"):
        recipe_from_dict(d)


def test_params_must_be_an_object():
    d = _good_dict()
    d["system"]["params"] = [1.0]
    with pytest.raises(ValueError, match=r"^system\.params: must be an obj"):
        recipe_from_dict(d)


def test_bad_object_value_names_the_field():
    d = _good_dict()
    d["system"]["params"]["omega"] = {"name": "J"}
    with pytest.raises(ValueError, match=r"^system\.params\.omega: "):
        recipe_from_dict(d)


def test_bad_param_name_type_raises():
    d = _good_dict()
    d["t_max"] = {"param": 3}
    with pytest.raises(ValueError, match=r"^t_max: "):
        recipe_from_dict(d)


def test_channels_must_be_a_list():
    d = _good_dict()
    d["channels"] = d["channels"][0]
    with pytest.raises(ValueError, match=r"^channels: must be a list"):
        recipe_from_dict(d)


def test_block_in_a_list_must_be_an_object():
    d = _good_dict()
    d["observe"][0] = "observe.purity"
    with pytest.raises(ValueError, match=r"^observe\[0\]: a block must be"):
        recipe_from_dict(d)


def test_modifier_first_is_rejected():
    d = _good_dict()
    d["channels"].insert(0, {"fn": "channels.ou_modulation", "params": {
        "s": 0.1, "tau": 1.0}})
    with pytest.raises(ValueError, match="before any channel producer"):
        recipe_from_dict(d)


def test_param_without_sampler_is_rejected():
    d = _good_dict()
    d["system"]["params"]["omega"] = {"param": "w"}
    with pytest.raises(ValueError, match="has no sampler"):
        recipe_from_dict(d)


@pytest.mark.parametrize("field, value", [
    ("t_max", -1.0), ("dt", "x"), ("seeds", "other"),
    ("legacy_rng_offset", 1.5)])
def test_bad_scalar_field_raises(field, value):
    d = _good_dict()
    d[field] = value
    with pytest.raises(ValueError, match="recipe: "):
        recipe_from_dict(d)


def test_missing_required_block_parameter_raises():
    d = _good_dict()
    del d["target"]["params"]["threshold"]
    with pytest.raises(ValueError, match="'threshold'"):
        recipe_from_dict(d)


# --- Round trip ---


def _sample_value(p):
    if "choices" in p:
        return p["choices"][0]
    return {"float": 0.5, "int": 2, "str": "a", "bool": False,
            "tuple": [0.5, 1.0]}[p["kind"]]


def _all_blocks():
    cat = catalog_lib.catalog()
    return [b for stage in cat["stages"].values() for b in stage]


@pytest.mark.parametrize("entry", _all_blocks(), ids=lambda b: b["fn"])
def test_every_block_kind_round_trips(entry):
    params = {p["name"]: _sample_value(p) for p in entry["params"]
              if p["injected"] is None}
    original = block(entry["fn"], **params)
    again = block_from_dict(_through_json(block_to_dict(original)), "x")
    assert again == original
    assert hash(again) == hash(original)


@pytest.mark.parametrize("entry", _all_blocks(), ids=lambda b: b["fn"])
def test_every_block_kind_round_trips_with_a_param(entry):
    names = [p["name"] for p in entry["params"] if p["kind"] == "float"
             and p["injected"] is None]
    if not names:
        pytest.skip("no float parameter")
    original = block(entry["fn"], **{names[0]: Param("v")})
    data = _through_json(block_to_dict(original))
    assert data["params"][names[0]] == {"param": "v"}
    assert block_from_dict(data, "x") == original


def _kondo():
    return recipe.Recipe(
        system=block("system.two_qubit", J=Param("J"), interaction="XXZ"),
        channels=(
            block("channels.kondo_bath", rho_J=0.1, kT=0.2, k_F=1.0, dim=2),
            block("channels.ou_modulation", s=0.3, tau=2.0),
            block("channels.bernstein_profile", coeffs=(1.0, 0.5, 0.2))),
        initial=block("initial.parity_sector", sector=1),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1"),
                 block("observe.gaussian_noise", sigma=0.01,
                       names=("purity", "coherence_l1"))),
        target=block("target.energy_basis_crossing", threshold=0.37),
        t_max=Param("T"), dt=0.5,
        sampler=block("samplers.uniform", J=(0.2, 1.0), T=(5.0, 9.0)),
        seeds="legacy", legacy_rng_offset=3)


def _record():
    return recipe.Recipe(
        system=block("system.single_qubit", omega=1.0),
        channels=(block("channels.jump", op_type="sigma_z", gamma=0.1),),
        initial=block("initial.random_pure"),
        evolve=block("evolve.sequential_record", basis="z"),
        observe=(block("observe.record_series"),),
        target=None, t_max=3.0, dt=0.5)


@pytest.mark.parametrize("make", [_kondo, _record])
def test_recipe_round_trips(make):
    original = make()
    data = _through_json(recipe_to_dict(original))
    assert recipe_from_dict(data) == original


def test_encoding_uses_lists_and_param_objects():
    data = recipe_to_dict(_kondo())
    assert data["t_max"] == {"param": "T"}
    assert data["system"]["params"]["J"] == {"param": "J"}
    assert data["sampler"]["params"]["J"] == [0.2, 1.0]
    assert data["channels"][2]["params"]["coeffs"] == [1.0, 0.5, 0.2]
    assert data["target"]["fn"] == "target.energy_basis_crossing"
    assert recipe_to_dict(_record())["target"] is None


def test_decoding_turns_lists_into_tuples():
    again = recipe_from_dict(_through_json(recipe_to_dict(_kondo())))
    assert again.sampler.param_dict()["J"] == (0.2, 1.0)


def test_defaults_are_filled_when_left_out():
    r = recipe_from_dict(_good_dict())
    assert r.evolve.fn == "evolve.master_equation"
    assert r.sampler is None and r.seeds == "spawn"


def test_input_dict_is_not_modified():
    d = _good_dict()
    before = copy.deepcopy(d)
    recipe_from_dict(d)
    assert d == before
