"""Scenario generators: Bernstein shapes, scenario groups, fixed coupling."""
import dataclasses

import numpy as np
import pytest

from src.data import scenarios
from src.domain import value_objects
from src.persistence import stores

NAN = float("nan")
INF = float("inf")
SHAPES = ("constant", "monotone_increasing", "monotone_decreasing", "peak",
          "step_up", "step_down", "random")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- bernstein_coeffs ------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_bernstein_coeffs_have_five_values_inside_the_rate_bounds(shape):
    for seed in range(20):
        coeffs = scenarios.bernstein_coeffs(shape, _rng(seed), gamma_max=0.8)
        assert len(coeffs) == 5 and isinstance(coeffs, tuple)
        assert all(0.05 <= c <= 0.8 for c in coeffs)


@pytest.mark.parametrize("shape", SHAPES)
def test_bernstein_coeffs_are_reproducible_for_equal_generators(shape):
    assert (scenarios.bernstein_coeffs(shape, _rng(3))
            == scenarios.bernstein_coeffs(shape, _rng(3)))


def test_bernstein_constant_shape_repeats_one_rate():
    coeffs = scenarios.bernstein_coeffs("constant", _rng())
    assert len(set(coeffs)) == 1


def test_bernstein_monotone_shapes_are_sorted():
    up = scenarios.bernstein_coeffs("monotone_increasing", _rng())
    down = scenarios.bernstein_coeffs("monotone_decreasing", _rng())
    assert list(up) == sorted(up) and list(down) == sorted(down, reverse=True)


def test_bernstein_peak_shape_is_symmetric_with_its_maximum_in_the_middle():
    for seed in range(10):
        c = scenarios.bernstein_coeffs("peak", _rng(seed))
        assert c[0] == c[4] and c[1] == c[3]
        assert c[2] == max(c) and c[0] == min(c)


def test_bernstein_steps_go_from_the_low_to_the_high_level_and_back():
    up = scenarios.bernstein_coeffs("step_up", _rng(1))
    down = scenarios.bernstein_coeffs("step_down", _rng(1))
    assert up[0] == up[1] < up[3] == up[4]
    assert up[2] == pytest.approx((up[0] + up[4]) / 2)
    assert down[0] == down[1] > down[3] == down[4]
    assert down == up[::-1]


def test_bernstein_a_lower_gamma_max_lowers_the_rates():
    for shape in SHAPES:
        coeffs = scenarios.bernstein_coeffs(shape, _rng(), gamma_max=0.2)
        assert max(coeffs) <= 0.2


@pytest.mark.parametrize("shape", ["constant", "random"])
def test_bernstein_smallest_gamma_max_gives_the_lower_edge(shape):
    coeffs = scenarios.bernstein_coeffs(shape, _rng(), gamma_max=0.05)
    assert set(coeffs) == {0.05}


def test_bernstein_edge_shapes_accept_the_smallest_workable_gamma_max():
    for shape in ("peak", "step_up", "step_down"):
        coeffs = scenarios.bernstein_coeffs(shape, _rng(), gamma_max=0.125)
        assert all(0.05 <= c <= 0.125 for c in coeffs)


# --- build_configs ---------------------------------------------------------

def test_build_configs_groups_follow_the_scenario_definitions():
    groups = scenarios.build_configs(n_per_scenario=4, seed=1)
    assert set(groups) == set(scenarios.SCENARIOS) | {"all"}
    assert all(len(groups[k]) == 4 for k in scenarios.SCENARIOS)
    assert groups["all"] == [c for k in "ABCDE" for c in groups[k]]
    a, b, c, d, e = (groups[k][0] for k in "ABCDE")
    assert a.dissipator.gamma_const is not None and a.t_max == 30.0
    assert b.dissipator.gamma_coeffs is not None and b.t_max == 20.0
    assert c.dissipator.operator_type is value_objects.DissipatorType.SIGMA_Z
    assert max(c.dissipator.gamma_coeffs) <= 0.2
    assert b.dissipator.operator_type is (
        value_objects.DissipatorType.SIGMA_MINUS)
    assert d.n_qubits is e.n_qubits is value_objects.QubitCount.TWO
    assert d.interaction_type is value_objects.InteractionType.XXZ
    assert e.interaction_type is value_objects.InteractionType.TFIM
    assert all(g.decoherence_criterion
               is value_objects.DecoherenceCriterion.COHERENCE
               for g in groups["all"])


def test_build_configs_parameters_are_inside_the_documented_ranges():
    groups = scenarios.build_configs(n_per_scenario=30, seed=2)
    for config in groups["all"]:
        assert 0.5 <= config.omega <= 2.0
    assert all(0.1 <= c.dissipator.gamma_const <= 0.8 for c in groups["A"])
    assert all(0.2 <= c.J <= 1.0 for c in groups["D"] + groups["E"])


def test_build_configs_cycles_through_the_gamma_shapes():
    groups = scenarios.build_configs(n_per_scenario=12, seed=2)
    # Shape 2 and 3 of the one-qubit cycle are monotone.
    inc = groups["B"][1].dissipator.gamma_coeffs
    dec = groups["B"][2].dissipator.gamma_coeffs
    assert list(inc) == sorted(inc) and list(dec) == sorted(dec, reverse=True)


def test_build_configs_same_seed_gives_equal_groups_and_other_seed_differs():
    a = scenarios.build_configs(3, seed=7)
    b = scenarios.build_configs(3, seed=7)
    c = scenarios.build_configs(3, seed=8)
    assert a == b and a != c


def test_build_configs_without_a_seed_still_builds_valid_groups():
    assert len(scenarios.build_configs(1, seed=None)["A"]) == 1


# --- fixed_coupling_configs ------------------------------------------------

def test_fixed_coupling_configs_share_the_coupling():
    cfgs = scenarios.fixed_coupling_configs(
        7, 1.3, np.random.default_rng(0),
        interaction=value_objects.InteractionType.XXZ)
    assert len(cfgs) == 7 and {c.J for c in cfgs} == {1.3}
    assert {c.interaction_type for c in cfgs} == {
        value_objects.InteractionType.XXZ}


def test_fixed_coupling_configs_default_to_tfim_and_the_coherence_criterion():
    cfgs = scenarios.fixed_coupling_configs(2, 0.5, _rng())
    assert {c.interaction_type for c in cfgs} == {
        value_objects.InteractionType.TFIM}
    assert {c.decoherence_criterion for c in cfgs} == {
        value_objects.DecoherenceCriterion.COHERENCE}
    assert {c.n_qubits for c in cfgs} == {value_objects.QubitCount.TWO}


def test_fixed_coupling_configs_take_the_requested_criterion():
    cfgs = scenarios.fixed_coupling_configs(
        2, 0.5, _rng(),
        criterion=value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH)
    assert {c.decoherence_criterion for c in cfgs} == {
        value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH}


def test_fixed_coupling_configs_cycle_through_six_gamma_shapes():
    cfgs = scenarios.fixed_coupling_configs(7, 0.5, _rng(4))
    assert len({c.dissipator.gamma_coeffs for c in cfgs}) == 7
    dec = cfgs[5].dissipator.gamma_coeffs
    assert list(dec) == sorted(dec, reverse=True)


def test_fixed_coupling_configs_draw_new_values_from_a_used_generator():
    rng = _rng(1)
    first = scenarios.fixed_coupling_configs(2, 0.5, rng)
    second = scenarios.fixed_coupling_configs(2, 0.5, rng)
    assert first != second
    assert first == scenarios.fixed_coupling_configs(2, 0.5, _rng(1))


# --- with_criterion and the store ------------------------------------------

def test_with_criterion_copies_the_configs_with_another_criterion():
    groups = scenarios.build_configs(2, seed=0)
    out = scenarios.with_criterion(
        groups["D"], value_objects.DecoherenceCriterion.CONCURRENCE)
    assert [c.decoherence_criterion for c in out] == [
        value_objects.DecoherenceCriterion.CONCURRENCE] * 2
    assert [dataclasses.replace(
        c, decoherence_criterion=value_objects.DecoherenceCriterion.COHERENCE
    ) for c in out] == groups["D"]
    assert groups["D"][0].decoherence_criterion is (
        value_objects.DecoherenceCriterion.COHERENCE)


def test_with_criterion_accepts_any_sequence_and_an_empty_one():
    groups = scenarios.build_configs(1, seed=0)
    purity = value_objects.DecoherenceCriterion.PURITY
    assert scenarios.with_criterion([], purity) == []
    out = scenarios.with_criterion(tuple(groups["all"]), purity)
    assert len(out) == 5 and isinstance(out, list)


def test_trajectory_store_is_the_shared_numpy_cache(isolated_cwd):
    store = scenarios.trajectory_store()
    assert isinstance(store, stores.NumpyTrajectoryStore)
    assert (isolated_cwd / "data" / "trajectories").is_dir()


def test_scenario_labels_are_a_to_e():
    assert list(scenarios.SCENARIOS) == ["A", "B", "C", "D", "E"]


# --- Negative cases ---


@pytest.mark.parametrize("shape", ["zigzag", "Peak", "", None, 3])
def test_bernstein_unknown_shape_raises_value_error(shape):
    with pytest.raises(ValueError, match="Unknown gamma"):
        scenarios.bernstein_coeffs(shape, _rng())


@pytest.mark.parametrize("gamma_max", [0.0, -1.0, 0.049, NAN, INF, -INF])
@pytest.mark.parametrize("shape", SHAPES)
def test_bernstein_gamma_max_below_the_lower_edge_raises_value_error(
        shape, gamma_max):
    with pytest.raises(ValueError, match="gamma_max must be finite"):
        scenarios.bernstein_coeffs(shape, _rng(), gamma_max=gamma_max)


@pytest.mark.parametrize("shape", ["peak", "step_up", "step_down"])
@pytest.mark.parametrize("gamma_max", [0.05, 0.1, 0.12])
def test_bernstein_edge_shape_with_a_small_gamma_max_raises_value_error(
        shape, gamma_max):
    with pytest.raises(ValueError, match="needs gamma_max of at least 0.125"):
        scenarios.bernstein_coeffs(shape, _rng(), gamma_max=gamma_max)


def test_bernstein_without_a_generator_raises_attribute_error():
    # Debatable, kept: no explicit check; the error names the missing method.
    with pytest.raises(AttributeError, match="uniform"):
        scenarios.bernstein_coeffs("random", None)


@pytest.mark.parametrize("n", [0, -1, -100])
def test_build_configs_fewer_than_one_configuration_raises_value_error(n):
    with pytest.raises(ValueError, match="n_per_scenario must be at least 1"):
        scenarios.build_configs(n_per_scenario=n, seed=0)


def test_build_configs_fractional_count_raises_type_error():
    with pytest.raises(TypeError, match="integer"):
        scenarios.build_configs(n_per_scenario=1.5, seed=0)


def test_build_configs_negative_seed_raises_value_error():
    with pytest.raises(ValueError, match="non-negative"):
        scenarios.build_configs(n_per_scenario=1, seed=-1)


@pytest.mark.parametrize("n", [0, -3])
def test_fixed_coupling_configs_fewer_than_one_configuration_raises(n):
    with pytest.raises(ValueError, match="n must be at least 1"):
        scenarios.fixed_coupling_configs(n, 0.5, _rng())


@pytest.mark.parametrize("coupling", [0.0, -1.0, NAN, INF])
def test_fixed_coupling_configs_invalid_coupling_raises_value_error(coupling):
    with pytest.raises(ValueError, match="J"):
        scenarios.fixed_coupling_configs(1, coupling, _rng())


@pytest.mark.parametrize("interaction", ["TFIM", "ZZ", None])
def test_fixed_coupling_configs_interaction_that_is_not_an_enum_raises(
        interaction):
    with pytest.raises(TypeError, match="interaction_type"):
        scenarios.fixed_coupling_configs(
            1, 0.5, _rng(), interaction=interaction)


@pytest.mark.parametrize("criterion", ["purity", "x", None])
def test_fixed_coupling_configs_criterion_that_is_not_an_enum_raises(
        criterion):
    with pytest.raises(TypeError, match="decoherence_criterion"):
        scenarios.fixed_coupling_configs(
            1, 0.5, _rng(), criterion=criterion)


def test_fixed_coupling_configs_without_a_generator_raises_attribute_error():
    # Debatable, kept: see bernstein_coeffs.
    with pytest.raises(AttributeError, match="uniform"):
        scenarios.fixed_coupling_configs(1, 0.5, None)


@pytest.mark.parametrize("criterion", [
    value_objects.DecoherenceCriterion.COHERENCE_ENERGY,
    value_objects.DecoherenceCriterion.CONCURRENCE,
    value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH])
def test_with_criterion_two_qubit_criterion_on_a_one_qubit_group_raises(
        criterion):
    groups = scenarios.build_configs(1, seed=0)
    with pytest.raises(ValueError, match="two qubits only"):
        scenarios.with_criterion(groups["A"], criterion)
    with pytest.raises(ValueError, match="two qubits only"):
        scenarios.with_criterion(groups["all"], criterion)


@pytest.mark.parametrize("criterion", ["purity", "garbage", None])
def test_with_criterion_that_is_not_an_enum_member_raises_type_error(
        criterion):
    groups = scenarios.build_configs(1, seed=0)
    with pytest.raises(TypeError, match="decoherence_criterion"):
        scenarios.with_criterion(groups["B"], criterion)
