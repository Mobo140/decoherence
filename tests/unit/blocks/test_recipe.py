"""blocks.recipe: Block, Param and Recipe validation."""
import dataclasses

import pytest

from src.blocks import recipe
from src.blocks.recipe import Param, block


def _good(**overrides):
    fields = dict(
        system=block("system.single_qubit", omega=1.0),
        channels=(block("channels.jump", op_type="sigma_minus", gamma=0.2),),
        initial=block("initial.random_pure"),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1")),
        target=block("target.coherence_crossing", threshold=0.5),
    )
    fields.update(overrides)
    return recipe.Recipe(**fields)


# --- Negative cases: block ---


@pytest.mark.parametrize("fn", [
    "jump", "channels.nope", "nomodule.jump", "channels._scaled", "", ".jump",
    "channels."])
def test_block_unresolvable_function_raises(fn):
    with pytest.raises(ValueError, match="fn"):
        block(fn)


def test_block_non_string_function_raises():
    with pytest.raises(ValueError, match="must be a string"):
        block(None)


def test_block_unknown_parameter_raises_and_lists_the_valid_ones():
    with pytest.raises(ValueError, match="no parameter 'rate'.*gamma"):
        block("channels.jump", rate=1.0)


@pytest.mark.parametrize("name", ["system", "channels", "states", "rng"])
def test_block_reserved_parameter_raises(name):
    with pytest.raises(ValueError, match="supplied by the runner"):
        block("initial.parity_sector", **{name: 1})


@pytest.mark.parametrize("value", [{"a": 1}, object(), [1, {"a": 2}]])
def test_block_unhashable_value_raises(value):
    with pytest.raises(ValueError, match="must be a number, string"):
        block("channels.jump", gamma=value)


def test_param_needs_a_name():
    with pytest.raises(ValueError, match="non-empty name"):
        Param("")


# --- Negative cases: recipe ---


def test_recipe_unknown_seed_mode_raises():
    with pytest.raises(ValueError, match="seeds must be one of"):
        _good(seeds="random")


@pytest.mark.parametrize("field", ["t_max", "dt"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), "1"])
def test_recipe_invalid_time_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} must be"):
        _good(**{field: value})


def test_recipe_missing_required_parameter_raises():
    with pytest.raises(ValueError, match="required parameter 'omega'"):
        _good(system=block("system.single_qubit"))


def test_recipe_channel_block_with_another_first_parameter_raises():
    with pytest.raises(ValueError, match="first parameter"):
        _good(channels=(block("recipe.freeze", value=1),))


def test_recipe_modifier_before_any_producer_raises():
    modifier = block("channels.bernstein_profile", coeffs=(1.0,), t_max=2.0)
    with pytest.raises(ValueError, match="before any channel producer"):
        _good(channels=(modifier,))


def test_recipe_without_channels_or_observations_raises():
    with pytest.raises(ValueError, match="at least one channel"):
        _good(channels=())
    with pytest.raises(ValueError, match="at least one channel"):
        _good(observe=())


def test_recipe_first_observation_block_must_be_a_producer():
    with pytest.raises(ValueError, match="must be a producer"):
        _good(observe=(block("observe.model_layout"),))


def test_recipe_params_without_a_sampler_raise():
    with pytest.raises(ValueError, match="no sampler"):
        _good(system=block("system.single_qubit", omega=Param("omega")))


def test_recipe_unsupplied_time_param_without_a_sampler_raises():
    with pytest.raises(ValueError, match="no sampler"):
        _good(t_max=Param("t_max"))


def test_recipe_sampler_with_params_raises():
    sampler = block("samplers.fixed", a=Param("a"))
    with pytest.raises(ValueError, match="cannot be Params"):
        _good(sampler=sampler)


def test_recipe_sampler_without_rng_and_n_raises():
    with pytest.raises(ValueError, match=r"\(rng, n"):
        _good(sampler=block("recipe.freeze"))


# --- Positive cases ---


def test_freeze_rejects_containers_of_unhashable_data():
    with pytest.raises(ValueError, match="must be a number"):
        recipe.freeze((1, [2, {"a": 3}]))
    assert recipe.freeze([1, [2, "a", None]]) == (1, (2, "a", None))


def test_block_is_frozen_hashable_and_sorts_parameters():
    a = block("channels.jump", op_type="sigma_z", gamma=0.1)
    b = block("channels.jump", gamma=0.1, op_type="sigma_z")
    assert a == b and hash(a) == hash(b)
    assert [k for k, _ in a.params] == ["gamma", "op_type"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.fn = "x"


def test_block_turns_lists_into_tuples_recursively():
    b = block("channels.bernstein_profile", coeffs=[0.1, [0.2, 0.3]],
              t_max=1.0)
    assert dict(b.params)["coeffs"] == (0.1, (0.2, 0.3))
    hash(b)


def test_block_accepts_params_as_values_and_reports_their_names():
    b = block("channels.jump", op_type=Param("op"), gamma=(Param("g"), 1))
    assert b.param_names() == {"op", "g"}


def test_block_accepts_keyword_catching_functions():
    assert block("samplers.fixed", anything=1).param_dict() == {"anything": 1}


def test_recipe_is_hashable_and_collects_params():
    r = _good(
        system=block("system.single_qubit", omega=Param("omega")),
        t_max=Param("t_max"),
        sampler=block("samplers.fixed", omega=1.0, t_max=2.0))
    hash(r)
    assert r.referenced_params() == {"omega", "t_max"}
    assert len(r.blocks()) == 8


def test_recipe_accepts_producers_and_modifiers_in_any_valid_order():
    _good(channels=(
        block("channels.jump", op_type="sigma_z", gamma=0.1),
        block("channels.ou_modulation", s=0.2, tau=1.0),
        block("channels.jump", op_type="sigma_minus", gamma=0.1)))


# --- evolution stage ---


def _sequential(**overrides):
    fields = dict(
        evolve=block("evolve.sequential_record", basis="z"),
        observe=(block("observe.record_series"),), target=None)
    fields.update(overrides)
    return _good(**fields)


def test_recipe_unknown_evolution_block_raises():
    with pytest.raises(ValueError, match="evolve must be one of"):
        _good(evolve=block("channels.jump", op_type="sigma_z", gamma=0.1))


def test_recipe_sequential_record_needs_its_basis():
    with pytest.raises(ValueError, match="required parameter 'basis'"):
        _sequential(evolve=block("evolve.sequential_record"))


def test_recipe_observe_blocks_must_match_the_evolution():
    with pytest.raises(ValueError, match="takes 'states' but"):
        _good(evolve=block("evolve.sequential_record", basis="z"))
    with pytest.raises(ValueError, match="takes 'record' but"):
        _good(observe=(block("observe.record_series"),))


def test_recipe_target_must_match_the_evolution():
    with pytest.raises(ValueError, match="required parameter 'states'"):
        _sequential(target=block("target.sudden_death"))


def test_recipe_defaults_to_the_master_equation():
    assert _good().evolve == block("evolve.master_equation")


def test_recipe_accepts_sequential_record_without_a_target():
    recipe_ = _sequential()
    assert recipe_.target is None
    hash(recipe_)
