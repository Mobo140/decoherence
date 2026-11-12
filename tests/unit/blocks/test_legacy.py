"""blocks.legacy: recipes of the published simulators."""
import dataclasses

import pytest

from src.blocks import legacy
from src.data.scenarios import build_configs
from src.domain import value_objects

NAN = float("nan")
GROUPS = build_configs(1, 0)
CONST_1Q = GROUPS["A"][0]
BERNSTEIN_1Q = GROUPS["B"][0]
XXZ = GROUPS["D"][0]


def _fns(recipe_blocks):
    return [b.fn for b in recipe_blocks]


# --- Negative cases ---


def test_unknown_alias_raises_key_error_listing_the_valid_ones():
    with pytest.raises(KeyError, match="plain"):
        legacy.recipe_for(XXZ, "nope")


@pytest.mark.parametrize("alias, params, match", [
    ("stochastic_gamma", {}, "missing"),
    ("stochastic_gamma", {"s": 0.3, "tau": 1.0}, "seed"),
    ("parity_sector", {}, "sector"),
    ("kondo", {"rho_J": 0.1, "kT": 0.1}, "kF_d"),
    ("plain", {"foo": 1}, "unexpected"),
])
def test_missing_or_unexpected_arguments_raise_type_error(
        alias, params, match):
    with pytest.raises(TypeError, match=match):
        legacy.validate_simulator(alias, params)


@pytest.mark.parametrize("s", [-0.1, NAN, float("inf")])
def test_stochastic_gamma_invalid_strength_raises(s):
    with pytest.raises(ValueError, match="s must be"):
        legacy.validate_simulator(
            "stochastic_gamma", {"s": s, "tau": 1.0, "seed": 0})


@pytest.mark.parametrize("tau", [0.0, -1.0, NAN, None])
def test_stochastic_gamma_invalid_correlation_time_raises(tau):
    with pytest.raises(ValueError, match="tau must be"):
        legacy.validate_simulator(
            "stochastic_gamma", {"s": 0.3, "tau": tau, "seed": 0})


def test_stochastic_gamma_negative_seed_raises():
    with pytest.raises(ValueError, match="non-negative"):
        legacy.validate_simulator(
            "stochastic_gamma", {"s": 0.3, "tau": 1.0, "seed": -1})


@pytest.mark.parametrize("sector", [0, 2, -2, None, "1", "+"])
def test_parity_sector_other_than_plus_or_minus_one_raises(sector):
    with pytest.raises(ValueError, match=r"sector must be \+1 or -1"):
        legacy.validate_simulator("parity_sector", {"sector": sector})


@pytest.mark.parametrize("override, match", [
    ({"kF_d": -1.0}, "kF_d"), ({"kF_d": NAN}, "kF_d"),
    ({"rho_J": 1.0}, "rho_J"), ({"kT": 0.0}, "kT"), ({"dim": 4}, "dim")])
def test_kondo_invalid_parameters_raise(override, match):
    params = dict(rho_J=0.1, kT=0.5, kF_d=1.0, dim=2)
    params.update(override)
    with pytest.raises(ValueError, match=match):
        legacy.validate_simulator("kondo", params)


# --- Positive cases ---


def test_parity_sector_accepts_equal_numeric_types():
    legacy.validate_simulator("parity_sector", {"sector": 1.0})
    legacy.validate_simulator("parity_sector", {"sector": True})


def test_plain_recipe_of_a_bernstein_config():
    recipe = legacy.recipe_for(BERNSTEIN_1Q)
    assert recipe.seeds == "legacy" and recipe.sampler is None
    assert _fns(recipe.channels) == [
        "channels.jump", "channels.bernstein_profile"]
    assert recipe.system.fn == "system.single_qubit"
    assert _fns(recipe.observe) == [
        "observe.pauli_means", "observe.purity", "observe.coherence_l1"]
    assert (recipe.t_max, recipe.dt) == (
        BERNSTEIN_1Q.t_max, BERNSTEIN_1Q.dt)


def test_plain_recipe_of_a_constant_config_has_one_jump_block():
    recipe = legacy.recipe_for(CONST_1Q)
    assert _fns(recipe.channels) == ["channels.jump"]
    assert dict(recipe.channels[0].params)["gamma"] == (
        CONST_1Q.dissipator.gamma_const)


@pytest.mark.parametrize("criterion, fn", [
    (value_objects.DecoherenceCriterion.COHERENCE,
     "target.coherence_crossing"),
    (value_objects.DecoherenceCriterion.PURITY,
     "target.coherence_crossing"),
    (value_objects.DecoherenceCriterion.CONCURRENCE,
     "target.coherence_crossing"),
    (value_objects.DecoherenceCriterion.COHERENCE_ENERGY,
     "target.energy_basis_crossing"),
    (value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH,
     "target.sudden_death")])
def test_the_criterion_selects_the_target_block(criterion, fn):
    config = dataclasses.replace(XXZ, decoherence_criterion=criterion)
    assert legacy.recipe_for(config).target.fn == fn


def test_extra_observables_only_for_two_qubits_and_before_the_tail():
    entanglement = legacy.recipe_for(XXZ, "entanglement")
    assert _fns(entanglement.observe) == [
        "observe.pauli_means", "observe.correlators", "observe.concurrence",
        "observe.purity", "observe.coherence_l1"]
    assert "observe.correlators" in _fns(
        legacy.recipe_for(XXZ, "correlators").observe)
    assert "observe.concurrence" not in _fns(
        legacy.recipe_for(XXZ, "correlators").observe)
    assert _fns(legacy.recipe_for(BERNSTEIN_1Q, "entanglement").observe) == (
        _fns(legacy.recipe_for(BERNSTEIN_1Q).observe))


def test_stochastic_gamma_leaves_a_constant_rate_unmodulated():
    params = dict(s=0.3, tau=1.0, seed=1)
    constant = legacy.recipe_for(CONST_1Q, "stochastic_gamma", **params)
    varying = legacy.recipe_for(BERNSTEIN_1Q, "stochastic_gamma", **params)
    assert dict(constant.channels[-1].params)["modulate_constant"] is False
    assert dict(varying.channels[-1].params)["modulate_constant"] is True
    assert "seed" not in dict(varying.channels[-1].params)


def test_parity_sector_falls_back_to_plain_for_one_qubit():
    assert legacy.recipe_for(XXZ, "parity_sector", sector=1).initial.fn == (
        "initial.parity_sector")
    assert legacy.recipe_for(
        BERNSTEIN_1Q, "parity_sector", sector=1).initial.fn == (
        "initial.random_pure")


def test_kondo_replaces_the_jump_channels_and_scales_by_kf_d():
    recipe = legacy.recipe_for(
        XXZ, "kondo", rho_J=0.1, kT=0.5, kF_d=2.5, dim=3)
    assert _fns(recipe.channels) == ["channels.kondo_bath"]
    assert dict(recipe.channels[0].params) == {
        "rho_J": 0.1, "kT": 0.5, "k_F": 1.0, "dim": 3, "spacing": 2.5}


def test_recipes_are_hashable_and_equal_for_equal_input():
    a = legacy.recipe_for(XXZ, "entanglement")
    b = legacy.recipe_for(XXZ, "entanglement")
    assert a == b and hash(a) == hash(b)
