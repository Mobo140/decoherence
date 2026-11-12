"""QuTipSimulator and its decoherence-time logic.

Every test runs at most one tiny simulation; datasets and training are
covered under tests/integration.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import qutip as qt

from src.data.scenarios import build_configs
from src.domain.value_objects import (
    DecoherenceCriterion,
    DissipatorConfig,
    DissipatorType,
    InteractionType,
    QubitCount,
    SystemConfig,
)
from src.physics import lindblad_systems
from src.simulation import simulator
from src.simulation.simulator import QuTipSimulator, compute_t_decoh

NAN = float("nan")


def _plus_state(n_qubits: int = 1, seed=None) -> qt.Qobj:
    """Equal superposition so initial coherence is maximal."""
    dim = 2 ** n_qubits
    ket = qt.Qobj(np.ones((dim, 1), dtype=complex) / np.sqrt(dim))
    return qt.ket2dm(ket)


def _const_sigma_minus(
    gamma: float, t_max: float, dt: float = 0.1
) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma, DissipatorType.SIGMA_MINUS),
        t_max=t_max,
        dt=dt,
    )


def _two_qubit_config(
    criterion: DecoherenceCriterion = DecoherenceCriterion.COHERENCE,
    t_max: float = 3.0,
) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.TWO, omega=1.0,
        dissipator=DissipatorConfig.constant(0.4), J=0.5,
        interaction_type=InteractionType.XXZ, t_max=t_max, dt=0.1,
        decoherence_criterion=criterion)


def _fix_initial_state(monkeypatch, rho: qt.Qobj) -> None:
    """Makes the simulator start every trajectory from `rho`."""
    monkeypatch.setattr(
        lindblad_systems, "sample_random_pure_state",
        lambda n_qubits=1, seed=None: rho)


def _bell_state() -> qt.Qobj:
    ket = (qt.basis([2, 2], [0, 0]) + qt.basis([2, 2], [1, 1])).unit()
    return qt.ket2dm(ket)


def _lopsided_state() -> qt.Qobj:
    """sqrt(0.2)|00> + sqrt(0.8)|11>: its entanglement dies at a finite time."""
    ket = (np.sqrt(0.2) * qt.basis([2, 2], [0, 0])
           + np.sqrt(0.8) * qt.basis([2, 2], [1, 1]))
    return qt.ket2dm(ket)


def _product_state() -> qt.Qobj:
    return qt.ket2dm(qt.basis([2, 2], [0, 1]))


# --- crossing helpers ------------------------------------------------------

def test_sample_initial_state_has_the_dimension_of_the_system():
    one = simulator.sample_initial_state(_const_sigma_minus(0.1, 1.0))
    two = simulator.sample_initial_state(_two_qubit_config())
    assert one.shape == (2, 2) and two.shape == (4, 4)


def test_sample_initial_state_follows_the_global_numpy_seed():
    config = _two_qubit_config()
    np.random.seed(4)
    first = simulator.sample_initial_state(config).full()
    np.random.seed(4)
    second = simulator.sample_initial_state(config).full()
    np.testing.assert_array_equal(first, second)


def test_interpolate_crossing_finds_the_linear_crossing_time():
    times = np.array([0.0, 1.0, 2.0])
    values = [1.0, 0.6, 0.2]
    assert simulator.interpolate_crossing(times, values, 2, 0.4) == (
        pytest.approx(1.5))
    assert simulator.interpolate_crossing(times, values, 1, 0.8) == (
        pytest.approx(0.5))


def test_interpolate_crossing_at_the_first_point_returns_the_start_time():
    assert simulator.interpolate_crossing(
        np.array([2.0, 3.0]), [0.1, 0.0], 0, 0.5) == 2.0


def test_interpolate_crossing_equal_neighbours_stay_inside_the_interval():
    times = np.array([0.0, 1.0, 2.0])
    crossing = simulator.interpolate_crossing(times, [1.0, 1.0, 1.0], 1, 0.5)
    assert 0.0 <= crossing <= 1.0


def test_first_crossing_interpolates_a_linear_decay():
    times = np.linspace(0.0, 1.0, 11)
    values = 1.0 - times
    time, censored = simulator.first_crossing(times, values, 0.35)
    assert time == pytest.approx(0.65, abs=1e-9) and censored is False


def test_first_crossing_never_reached_is_censored_at_the_last_time():
    times = np.linspace(0.0, 2.0, 5)
    assert simulator.first_crossing(times, np.ones(5), 0.5) == (2.0, True)


def test_first_crossing_curve_that_starts_below_the_level_gives_the_start():
    times = np.linspace(1.0, 2.0, 5)
    assert simulator.first_crossing(times, np.full(5, 0.1), 0.5) == (
        1.0, False)


def test_first_crossing_equality_counts_only_when_inclusive():
    times = np.arange(4.0)
    values = [1.0, 0.5, 0.8, 0.2]       # touches 0.5, rises, then falls
    exclusive = simulator.first_crossing(times, values, 0.5)
    inclusive = simulator.first_crossing(times, values, 0.5, inclusive=True)
    assert exclusive == (pytest.approx(2.5), False)
    assert inclusive == (pytest.approx(1.0), False)


def test_first_crossing_accepts_plain_lists():
    assert simulator.first_crossing([0.0, 1.0, 2.0], [1.0, 0.0, 0.0], 0.5)[
        0] == pytest.approx(0.5)


def test_first_crossing_takes_the_first_of_several_crossings():
    times = np.arange(5.0)
    values = [1.0, 0.2, 0.9, 0.1, 0.0]
    assert simulator.first_crossing(times, values, 0.5)[0] < 1.0


def test_relative_level_scales_the_threshold_by_the_initial_value():
    assert simulator.relative_level([2.0, 1.0], 0.25) == pytest.approx(0.5)


def test_relative_level_of_a_curve_that_starts_at_zero_is_absolute():
    assert simulator.relative_level([0.0, 1.0], 0.3) == 0.3
    assert simulator.relative_level([1e-11, 1.0], 0.3) == 0.3
    assert simulator.relative_level([], 0.3) == 0.3


def test_compute_t_decoh_interpolates_synthetic_exponential():
    """Crossing of C0/e on C(t)=C0·exp(-γt/2) must recover T₂=2/γ."""
    gamma = 0.4
    t2 = 2.0 / gamma  # 5.0
    times = np.linspace(0.0, 30.0, 301)  # dt = 0.1
    c0 = 1.0
    coh = c0 * np.exp(-gamma * times / 2.0)
    cfg = _const_sigma_minus(gamma, t_max=30.0)
    t_decoh, censored = compute_t_decoh(
        cfg, times, {"coherence_l1": coh},
    )
    assert censored is False
    assert abs(t_decoh - t2) < 0.01


def test_compute_t_decoh_purity_measures_the_decay_towards_the_mixed_value():
    config = dataclasses.replace(
        _const_sigma_minus(0.4, 30.0),
        decoherence_criterion=DecoherenceCriterion.PURITY)
    times = np.linspace(0.0, 30.0, 301)
    # Purity 1 -> 1/2 as 1/2 + exp(-t/5)/2: the level is 1/e of the way
    # down, 1 - (1/2)/e, and the curve reaches it at t = 5 ln(1 / (1 - 1/e)).
    purity = 0.5 + 0.5 * np.exp(-times / 5.0)
    t_decoh, censored = compute_t_decoh(config, times, {"purity": purity})
    assert censored is False
    assert t_decoh == pytest.approx(5.0 * np.log(1 / (1 - 1 / np.e)), abs=0.01)


def test_compute_t_decoh_purity_of_two_qubits_decays_towards_a_quarter():
    config = dataclasses.replace(
        _two_qubit_config(), decoherence_criterion=DecoherenceCriterion.PURITY)
    times = np.linspace(0.0, 3.0, 31)
    purity = np.full(31, 0.9)
    # 0.9 never falls to 0.9 - (1 - 1/e) * 0.65.
    assert compute_t_decoh(config, times, {"purity": purity}) == (3.0, True)
    purity[10:] = 0.4
    t_decoh, censored = compute_t_decoh(config, times, {"purity": purity})
    assert censored is False and 0.9 <= t_decoh <= 1.0


def test_compute_t_decoh_coherence_that_starts_at_zero_uses_an_absolute_level():
    config = _const_sigma_minus(0.4, 3.0)
    times = np.linspace(0.0, 3.0, 31)
    assert compute_t_decoh(
        config, times, {"coherence_l1": np.zeros(31)}) == (0.0, False)


def test_energy_basis_t_decoh_follows_the_coherence_in_the_energy_basis():
    config = _two_qubit_config(DecoherenceCriterion.COHERENCE_ENERGY)
    times = np.linspace(0.0, 4.0, 41)
    rhos = []
    for t in times:
        c = 0.5 * np.exp(-t)
        rhos.append(np.array([[0.5, c], [c, 0.5]], dtype=complex))
    t_decoh, censored = simulator.energy_basis_t_decoh(
        config, times, np.eye(2), rhos)
    assert censored is False and t_decoh == pytest.approx(1.0, abs=0.01)


def test_energy_basis_t_decoh_accepts_qutip_density_matrices():
    config = _two_qubit_config(DecoherenceCriterion.COHERENCE_ENERGY)
    times = np.linspace(0.0, 1.0, 5)
    rhos = [qt.Qobj(0.5 * np.exp(-t) * qt.sigmax().full()
                    + 0.5 * np.eye(2)) for t in times]
    arrays = [r.full() for r in rhos]
    assert simulator.energy_basis_t_decoh(config, times, np.eye(2), rhos) == (
        simulator.energy_basis_t_decoh(config, times, np.eye(2), arrays))


# --- simulation ------------------------------------------------------------

def test_simulate_recovers_the_analytical_t2(monkeypatch):
    """γ=0.4, σ₋ → T₂ = 2/γ = 5.0 within 0.05 after interpolation."""
    monkeypatch.setattr(
        lindblad_systems, "sample_random_pure_state", _plus_state
    )
    traj = QuTipSimulator().simulate(_const_sigma_minus(0.4, t_max=30.0))
    assert traj.censored is False
    assert abs(traj.t_decoh - 5.0) < 0.05


def test_simulate_slow_decay_is_censored_at_t_max(monkeypatch):
    """γ=0.01, t_max=5 → T₂=200, threshold never crossed."""
    monkeypatch.setattr(
        lindblad_systems, "sample_random_pure_state", _plus_state
    )
    traj = QuTipSimulator().simulate(_const_sigma_minus(0.01, t_max=5.0))
    assert traj.censored is True
    assert traj.t_decoh == pytest.approx(5.0, abs=1e-9)


def test_simulate_drops_the_density_matrices_and_keeps_the_time_axis(
        monkeypatch):
    _fix_initial_state(monkeypatch, _plus_state())
    traj = QuTipSimulator().simulate(_const_sigma_minus(0.4, t_max=1.0))
    assert "density_matrices" not in traj.observables
    assert len(traj.times) == 11 and traj.times[-1] == 1.0
    assert traj.feature_names == [
        "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"]
    assert traj.system_config.t_max == 1.0


def test_simulate_one_qubit_purity_criterion_gives_a_crossing(monkeypatch):
    _fix_initial_state(monkeypatch, _plus_state())
    # Dephasing takes the purity of |+> from 1 towards 1/2.
    config = SystemConfig(
        n_qubits=QubitCount.ONE, omega=1.0,
        dissipator=DissipatorConfig.constant(0.8, DissipatorType.SIGMA_Z),
        t_max=2.0, dt=0.05,
        decoherence_criterion=DecoherenceCriterion.PURITY)
    traj = QuTipSimulator().simulate(config)
    assert traj.censored is False
    assert traj.t_decoh == pytest.approx(0.143, abs=0.02)


def test_simulate_two_qubits_records_the_pauli_expectations(monkeypatch):
    _fix_initial_state(monkeypatch, _bell_state())
    traj = QuTipSimulator().simulate(_two_qubit_config())
    assert traj.feature_names == [
        "sigma_x1", "sigma_y1", "sigma_z1", "sigma_x2", "sigma_y2",
        "sigma_z2", "purity", "coherence_l1"]
    assert "density_matrices" not in traj.observables


def test_simulate_two_qubit_coherence_criterion_uses_the_default_logic(
        monkeypatch):
    _fix_initial_state(monkeypatch, _bell_state())
    traj = QuTipSimulator().simulate(_two_qubit_config(t_max=1.0))
    expected = compute_t_decoh(
        traj.system_config, traj.times, traj.observables)
    assert (traj.t_decoh, traj.censored) == expected


@pytest.mark.parametrize("criterion", [
    DecoherenceCriterion.COHERENCE_ENERGY, DecoherenceCriterion.CONCURRENCE])
def test_simulate_entangled_state_has_a_finite_time_for_two_qubit_criteria(
        monkeypatch, criterion):
    _fix_initial_state(monkeypatch, _bell_state())
    traj = QuTipSimulator().simulate(_two_qubit_config(criterion, t_max=5.0))
    assert traj.censored is False and 0.0 < traj.t_decoh < 5.0


def test_simulate_separable_start_has_zero_sudden_death_time(monkeypatch):
    _fix_initial_state(monkeypatch, _product_state())
    config = _two_qubit_config(
        DecoherenceCriterion.ENTANGLEMENT_DEATH, t_max=1.0)
    traj = QuTipSimulator().simulate(config)
    assert (traj.t_decoh, traj.censored) == (0.0, False)


def test_simulate_lopsided_state_loses_its_entanglement_at_a_finite_time(
        monkeypatch):
    _fix_initial_state(monkeypatch, _lopsided_state())
    config = _two_qubit_config(
        DecoherenceCriterion.ENTANGLEMENT_DEATH, t_max=5.0)
    traj = QuTipSimulator().simulate(config)
    assert traj.censored is False
    assert traj.t_decoh == pytest.approx(1.73, abs=0.05)


def test_simulate_bell_state_keeps_its_entanglement_and_is_censored(
        monkeypatch):
    _fix_initial_state(monkeypatch, _bell_state())
    config = _two_qubit_config(
        DecoherenceCriterion.ENTANGLEMENT_DEATH, t_max=3.0)
    traj = QuTipSimulator().simulate(config)
    assert (traj.t_decoh, traj.censored) == (3.0, True)


def test_sudden_death_time_interpolates_the_zero_of_lambda():
    t = np.linspace(0.0, 1.0, 11)
    lam = 0.5 - t                                    # zero at t = 0.5
    assert abs(simulator.sudden_death_time(t, lam)[0] - 0.5) < 1e-12
    # Separable.
    assert simulator.sudden_death_time(t, -t - 0.1) == (0.0, False)
    # Censored.
    assert simulator.sudden_death_time(t, 1.0 + t) == (1.0, True)


def test_simulate_esd_criterion_changes_only_the_target():
    cfg = dataclasses.replace(
        build_configs(n_per_scenario=2, seed=3)["E"][0],
        decoherence_criterion=DecoherenceCriterion.ENTANGLEMENT_DEATH)
    np.random.seed(11)
    traj = simulator.QuTipSimulator().simulate(cfg)
    np.random.seed(11)
    plain = simulator.QuTipSimulator().simulate(dataclasses.replace(
        cfg, decoherence_criterion=DecoherenceCriterion.COHERENCE))
    # Same dynamics and observables; only the target differs.
    np.testing.assert_array_equal(
        traj.observables["coherence_l1"], plain.observables["coherence_l1"])
    assert traj.t_decoh != plain.t_decoh


def test_simulate_hooks_default_to_the_published_behaviour():
    sim = QuTipSimulator()
    config = _two_qubit_config()
    system = object()
    assert sim._prepare_system(system, config) is system
    assert sim._extra_observables([np.eye(4)]) == {}
    assert sim._initial_state(config).shape == (4, 4)


def test_simulate_extra_observables_come_before_purity_and_coherence(
        monkeypatch):
    class Extra(QuTipSimulator):
        def _extra_observables(self, density_matrices):
            return {"extra": np.arange(len(density_matrices), dtype=float)}

    _fix_initial_state(monkeypatch, _bell_state())
    traj = Extra().simulate(_two_qubit_config(t_max=0.5))
    assert traj.feature_names[-3:] == ["extra", "purity", "coherence_l1"]
    assert "density_matrices" not in traj.observables
    np.testing.assert_array_equal(traj.observables["extra"], np.arange(6))


def test_simulate_extra_observables_are_not_computed_for_one_qubit(
        monkeypatch):
    class Extra(QuTipSimulator):
        def _extra_observables(self, density_matrices):
            raise AssertionError("one-qubit runs have no extra observables")

    _fix_initial_state(monkeypatch, _plus_state())
    traj = Extra().simulate(_const_sigma_minus(0.4, t_max=0.5))
    assert "extra" not in traj.observables


# --- Negative cases ---


def test_first_crossing_empty_input_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        simulator.first_crossing(np.array([]), [], 0.5)
    with pytest.raises(ValueError, match="must not be empty"):
        simulator.first_crossing(np.arange(3.0), [], 0.5)
    with pytest.raises(ValueError, match="must not be empty"):
        simulator.first_crossing(np.array([]), [1.0, 0.0], 0.5)


def test_first_crossing_times_and_values_of_different_length_raise():
    with pytest.raises(ValueError, match="times has 5 points but values has 3"):
        simulator.first_crossing(np.arange(5.0), [1.0, 0.5, 0.0], 0.5)
    with pytest.raises(ValueError, match="times has 3 points but values has 5"):
        simulator.first_crossing(np.arange(3.0), np.linspace(1, 0, 5), 0.5)


@pytest.mark.parametrize("values", [
    [NAN, NAN, NAN], [1.0, NAN, 0.2], [1.0, 0.8, NAN]])
def test_first_crossing_nan_in_the_curve_raises_value_error(values):
    with pytest.raises(ValueError, match="NaN"):
        simulator.first_crossing(np.arange(3.0), values, 0.5)


def test_first_crossing_nan_level_raises_value_error():
    with pytest.raises(ValueError, match="NaN"):
        simulator.first_crossing(np.arange(3.0), [1.0, 0.5, 0.0], NAN)


def test_first_crossing_infinite_values_are_not_an_error():
    time, censored = simulator.first_crossing(
        np.arange(3.0), [float("inf"), 1.0, -float("inf")], 0.5)
    assert censored is False and 1.0 <= time <= 2.0


@pytest.mark.parametrize("index", [-1, 3, 20])
def test_interpolate_crossing_index_outside_the_curve_raises_value_error(
        index):
    with pytest.raises(ValueError, match="outside"):
        simulator.interpolate_crossing(
            np.arange(3.0), [1.0, 0.5, 0.0], index, 0.25)


def test_interpolate_crossing_values_shorter_than_times_raises():
    with pytest.raises(ValueError, match="outside"):
        simulator.interpolate_crossing(np.arange(5.0), [1.0, 0.0], 3, 0.5)


def test_sudden_death_time_empty_curve_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        simulator.sudden_death_time(np.array([]), [])


def test_sudden_death_time_nan_lambda_raises_value_error():
    with pytest.raises(ValueError, match="NaN"):
        simulator.sudden_death_time(np.arange(3.0), [0.5, NAN, -0.1])


def test_relative_level_does_not_check_the_initial_value():
    # Debatable, kept: a NaN or negative initial value falls back to the
    # absolute level instead of raising (first_crossing rejects NaN curves).
    assert simulator.relative_level([NAN, 1.0], 0.3) == 0.3
    assert simulator.relative_level([-1.0, 1.0], 0.3) == 0.3


def test_compute_t_decoh_missing_coherence_raises_key_error():
    with pytest.raises(KeyError, match="coherence_l1"):
        compute_t_decoh(
            _const_sigma_minus(0.4, 3.0), np.arange(3.0),
            {"purity": np.ones(3)})


def test_compute_t_decoh_missing_purity_raises_key_error():
    config = dataclasses.replace(
        _const_sigma_minus(0.4, 3.0),
        decoherence_criterion=DecoherenceCriterion.PURITY)
    with pytest.raises(KeyError, match="purity"):
        compute_t_decoh(config, np.arange(3.0), {"coherence_l1": np.ones(3)})


@pytest.mark.parametrize("criterion", [
    DecoherenceCriterion.COHERENCE_ENERGY, DecoherenceCriterion.CONCURRENCE,
    DecoherenceCriterion.ENTANGLEMENT_DEATH])
def test_compute_t_decoh_criterion_it_cannot_evaluate_raises_value_error(
        criterion):
    config = _two_qubit_config(criterion)
    observables = {"coherence_l1": np.ones(3), "purity": np.ones(3)}
    with pytest.raises(ValueError, match=criterion.value):
        compute_t_decoh(config, np.arange(3.0), observables)


def test_compute_t_decoh_empty_curves_raise_value_error():
    config = _const_sigma_minus(0.4, 3.0)
    with pytest.raises(ValueError, match="must not be empty"):
        compute_t_decoh(config, np.array([]), {"coherence_l1": np.array([])})


def test_compute_t_decoh_nan_curve_raises_value_error():
    with pytest.raises(ValueError, match="NaN"):
        compute_t_decoh(
            _const_sigma_minus(0.4, 3.0), np.arange(3.0),
            {"coherence_l1": np.array([1.0, NAN, 0.1])})


def test_energy_basis_t_decoh_non_hermitian_hamiltonian_raises():
    config = _two_qubit_config(DecoherenceCriterion.COHERENCE_ENERGY)
    with pytest.raises(ValueError, match="Hermitian"):
        simulator.energy_basis_t_decoh(
            config, np.arange(2.0), np.array([[0.0, 1.0], [0.0, 0.0]]),
            [np.eye(2) / 2] * 2)


@pytest.mark.parametrize("hamiltonian", [np.zeros((2, 3)), np.zeros(3)])
def test_energy_basis_t_decoh_non_square_hamiltonian_raises(hamiltonian):
    config = _two_qubit_config(DecoherenceCriterion.COHERENCE_ENERGY)
    with pytest.raises(ValueError, match="Hermitian square"):
        simulator.energy_basis_t_decoh(
            config, np.arange(2.0), hamiltonian, [np.eye(2) / 2] * 2)


def test_energy_basis_t_decoh_state_of_another_size_raises_value_error():
    config = _two_qubit_config(DecoherenceCriterion.COHERENCE_ENERGY)
    with pytest.raises(ValueError, match="eigenvectors have shape"):
        simulator.energy_basis_t_decoh(
            config, np.arange(2.0), np.eye(4), [np.eye(2) / 2] * 2)


def test_energy_basis_t_decoh_without_states_raises_value_error():
    config = _two_qubit_config(DecoherenceCriterion.COHERENCE_ENERGY)
    with pytest.raises(ValueError, match="must not be empty"):
        simulator.energy_basis_t_decoh(config, np.array([]), np.eye(2), [])


@pytest.mark.parametrize("criterion", [
    DecoherenceCriterion.COHERENCE_ENERGY, DecoherenceCriterion.CONCURRENCE,
    DecoherenceCriterion.ENTANGLEMENT_DEATH])
def test_one_qubit_system_with_a_two_qubit_criterion_cannot_be_simulated(
        criterion):
    # The configuration is rejected before any simulation can start, instead
    # of the simulator returning t_decoh = t_max, censored.
    with pytest.raises(ValueError, match="two qubits only"):
        dataclasses.replace(
            _const_sigma_minus(0.4, 3.0), decoherence_criterion=criterion)


@pytest.mark.parametrize("config", [None, "config", 3, {"t_max": 1.0}])
def test_simulate_non_config_raises_type_error(config):
    with pytest.raises(TypeError, match="SystemConfig"):
        QuTipSimulator().simulate(config)
