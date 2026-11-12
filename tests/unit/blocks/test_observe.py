"""blocks.observe: observations, combination, selection and noise."""
import numpy as np
import pytest
import qutip as qt

from src.blocks import channels, evolve, observe, system
from src.data import dataset as dataset_lib
from src.data import observation, windows
from src.physics import measures, simulate_qutip
from src.physics import observables as state_observables

NAN = float("nan")
INF = float("inf")
TIMES = np.linspace(0.0, 3.0, 7)


def _states(n_qubits):
    if n_qubits == 1:
        sys_ = system.single_qubit(1.0)
        rho0 = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
    else:
        sys_ = system.two_qubit(0.8, "XXZ")
        rho0 = qt.ket2dm((qt.basis([2, 2], [0, 1])
                          + qt.basis([2, 2], [1, 0])).unit())
    return evolve.master_equation(
        sys_, channels.jump(sys_, "sigma_minus", 0.2), rho0, TIMES)


def _obs(names=("a", "b", "c")):
    rng = np.random.default_rng(0)
    return observe.Observations(
        TIMES, {n: rng.standard_normal(TIMES.size) for n in names})


def _dataset(obs):
    """A one-window dataset holding the series as columns."""
    window = np.stack(list(obs.series.values()), axis=1)[None].astype(
        np.float32)
    return dataset_lib.Dataset(
        sequences=window, t_obs=np.zeros(1), t_decoh_abs=np.ones(1),
        remaining_time=np.ones(1), risk_labels=np.zeros(1),
        train_idx=np.array([0]), val_idx=np.array([], dtype=int),
        test_idx=np.array([], dtype=int),
        feature_names=list(obs.series), metadata={})


# --- Negative cases ---


def test_observations_with_a_series_of_the_wrong_length_raise():
    with pytest.raises(ValueError, match="Series 'a' has shape"):
        observe.Observations(TIMES, {"a": np.zeros(3)})


def test_observations_with_two_dimensional_times_raise():
    with pytest.raises(ValueError, match="times must be 1-D"):
        observe.Observations(np.zeros((2, 2)))


def test_exact_observations_of_empty_states_raise():
    empty = evolve.States(np.zeros(0), [])
    for function in (observe.pauli_means, observe.purity,
                     observe.coherence_l1, observe.concurrence,
                     observe.correlators):
        with pytest.raises(ValueError, match="at least one"):
            function(empty)


def test_states_with_mismatched_times_raise():
    states = evolve.States(TIMES[:3], _states(1).rho)
    with pytest.raises(ValueError, match="matrices for"):
        observe.purity(states)


def test_pauli_means_of_other_sizes_raise():
    states = evolve.States(np.zeros(1), [np.eye(3) / 3])
    with pytest.raises(ValueError, match="2x2 or 4x4"):
        observe.pauli_means(states)


@pytest.mark.parametrize("function", [observe.concurrence,
                                      observe.correlators])
def test_two_qubit_observations_of_one_qubit_states_raise(function):
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        function(_states(1))


def test_combine_without_arguments_raises():
    with pytest.raises(ValueError, match="at least one"):
        observe.combine()


def test_combine_with_a_repeated_name_raises():
    with pytest.raises(ValueError, match=r"Duplicate series names: \['a'\]"):
        observe.combine(_obs(["a"]), _obs(["a", "b"]))


def test_combine_with_different_time_grids_raises():
    other = observe.Observations(TIMES + 1.0, {"z": np.zeros(TIMES.size)})
    with pytest.raises(ValueError, match="one time grid"):
        observe.combine(_obs(), other)


def test_select_unknown_name_raises_and_lists_the_available():
    with pytest.raises(ValueError, match=r"Unknown series \['z'\].*'a'"):
        observe.select(_obs(), ["a", "z"])


def test_select_single_string_raises_type_error():
    with pytest.raises(TypeError, match="collection of names"):
        observe.select(_obs(), "a")


def test_select_nothing_raises():
    with pytest.raises(ValueError, match="at least one name"):
        observe.select(_obs(), [])


@pytest.mark.parametrize("missing", observe.MODEL_TAIL)
def test_model_layout_without_a_tail_series_raises(missing):
    names = [n for n in ("a", *observe.MODEL_TAIL) if n != missing]
    with pytest.raises(ValueError, match=f"needs the series.*{missing}"):
        observe.model_layout(_obs(names))


@pytest.mark.parametrize("sigma", [-0.1, NAN, INF])
def test_noise_with_invalid_sigma_raises(sigma):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="sigma must be"):
        observe.gaussian_noise(_obs(), sigma, rng)
    with pytest.raises(ValueError, match="sigma must be"):
        observe.ar1_noise(_obs(), sigma, 0.5, rng)


def test_noise_with_a_non_generator_raises_type_error():
    with pytest.raises(TypeError, match="Generator"):
        observe.gaussian_noise(_obs(), 0.1, 3)
    with pytest.raises(TypeError, match="Generator"):
        observe.ar1_noise(_obs(), 0.1, 0.5, None)


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.5, NAN])
def test_ar1_noise_with_phi_outside_the_unit_interval_raises(phi):
    with pytest.raises(ValueError, match="phi"):
        observe.ar1_noise(_obs(), 0.1, phi, np.random.default_rng(0))


def test_noise_on_an_unknown_series_raises():
    with pytest.raises(ValueError, match="Unknown series"):
        observe.gaussian_noise(
            _obs(), 0.1, np.random.default_rng(0), names=["z"])


# --- Positive cases: exact observations ---


def test_one_qubit_series_equal_the_legacy_simulation_observables():
    states = _states(1)
    legacy = simulate_qutip.trajectories(
        states.rho, ["sigma_x", "sigma_y", "sigma_z"],
        state_observables.compute_observables_single_qubit)
    obs = observe.combine(
        observe.pauli_means(states), observe.purity(states),
        observe.coherence_l1(states))
    assert list(obs.series) == [
        "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"]
    for name, values in obs.series.items():
        np.testing.assert_array_equal(values, legacy[name])
    np.testing.assert_array_equal(obs.times, states.times)


def test_two_qubit_series_equal_the_legacy_simulation_observables():
    states = _states(2)
    names = [f"sigma_{a}{q}" for q in "12" for a in "xyz"]
    legacy = simulate_qutip.trajectories(
        states.rho, names,
        state_observables.compute_observables_two_qubit)
    obs = observe.combine(
        observe.pauli_means(states), observe.purity(states),
        observe.coherence_l1(states))
    assert list(obs.series) == names + ["purity", "coherence_l1"]
    for name, values in obs.series.items():
        np.testing.assert_array_equal(values, legacy[name])


def test_correlators_and_concurrence_equal_their_definitions():
    states = _states(2)
    series = observe.correlators(states).series
    assert list(series) == list(observe.CORR)
    np.testing.assert_array_equal(
        observe.concurrence(states).series["concurrence"],
        [measures.concurrence(r) for r in states.rho])
    for name, op in observe.CORR.items():
        np.testing.assert_allclose(
            series[name], [np.real(np.trace(r @ op)) for r in states.rho],
            atol=1e-12)


def test_correlators_of_known_states():
    up_down = qt.ket2dm(qt.basis([2, 2], [0, 1])).full()
    states = evolve.States(np.zeros(1), [up_down])
    series = observe.correlators(states).series
    assert series["corr_zz"][0] == pytest.approx(-1.0)
    assert series["corr_xx"][0] == pytest.approx(0.0)


# --- Positive cases: combine and select ---


def test_combine_keeps_the_argument_order():
    obs = observe.combine(_obs(["z"]), _obs(["a", "m"]))
    assert list(obs.series) == ["z", "a", "m"]


def test_select_keeps_the_order_of_the_names():
    obs = observe.select(_obs(), ["c", "a"])
    assert list(obs.series) == ["c", "a"]
    np.testing.assert_array_equal(obs.series["a"], _obs().series["a"])


def test_model_layout_moves_the_tail_to_the_end_keeping_the_rest():
    obs = _obs(["coherence_l1", "a", "purity", "b"])
    out = observe.model_layout(obs)
    assert list(out.series) == ["a", "b", "purity", "coherence_l1"]
    np.testing.assert_array_equal(
        out.series["a"], obs.series["a"])


def test_model_layout_of_only_the_tail_orders_it():
    out = observe.model_layout(_obs(["coherence_l1", "purity"]))
    assert list(out.series) == list(observe.MODEL_TAIL)


def test_model_tail_is_the_layout_of_the_padding():
    assert windows.PHYSICS_COLUMNS == len(observe.MODEL_TAIL)
    assert observe.MODEL_TAIL == ("purity", "coherence_l1")


def test_the_ar1_filter_of_the_data_package_is_the_block_function():
    assert observation.ar1_filter is observe.ar1_filter


# --- Positive cases: noise ---


def test_gaussian_noise_follows_the_legacy_draw_layout():
    obs = _obs()
    noisy = observe.gaussian_noise(obs, 0.1, np.random.default_rng(5))
    legacy = observation.gaussian(
        _dataset(obs), sigma=0.1, rng=np.random.default_rng(5)).sequences[0]
    got = np.stack(list(noisy.series.values()), axis=1)
    np.testing.assert_allclose(got, legacy, atol=1e-6)
    assert not np.allclose(got, np.stack(list(obs.series.values()), 1))


def test_gaussian_noise_zero_sigma_returns_the_input_and_draws_nothing():
    rng = np.random.default_rng(5)
    obs = _obs()
    assert observe.gaussian_noise(obs, 0.0, rng) is obs
    assert rng.normal() == np.random.default_rng(5).normal()


def test_ar1_noise_follows_the_legacy_filter():
    obs = _obs()
    noisy = observe.ar1_noise(obs, 0.2, 0.7, np.random.default_rng(6))
    legacy = observation.ar1(
        _dataset(obs), sigma=0.2, phi=0.7,
        rng=np.random.default_rng(6)).sequences[0]
    np.testing.assert_allclose(
        np.stack(list(noisy.series.values()), axis=1), legacy, atol=1e-6)


def test_noise_perturbs_only_the_named_series():
    obs = _obs()
    noisy = observe.gaussian_noise(
        obs, 0.1, np.random.default_rng(1), names=["b"])
    np.testing.assert_array_equal(noisy.series["a"], obs.series["a"])
    assert not np.array_equal(noisy.series["b"], obs.series["b"])
    assert list(noisy.series) == list(obs.series)


def test_noise_does_not_modify_its_input():
    obs = _obs()
    before = {k: v.copy() for k, v in obs.series.items()}
    observe.ar1_noise(obs, 0.3, 0.5, np.random.default_rng(1))
    for name, values in before.items():
        np.testing.assert_array_equal(obs.series[name], values)
