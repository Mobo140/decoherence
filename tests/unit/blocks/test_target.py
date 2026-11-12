"""blocks.target: decoherence times from the states."""
import dataclasses
import math

import numpy as np
import pytest
import qutip as qt

from src.blocks import channels, evolve, initial, system, target
from src.domain.value_objects import (
    DecoherenceCriterion,
    DissipatorConfig,
    InteractionType,
    QubitCount,
    SystemConfig,
)
from src.physics import measures
from src.simulation import simulator

NAN = float("nan")
ONE = system.single_qubit(1.0)
TWO = system.two_qubit(0.8, "XXZ")
TIMES = np.linspace(0.0, 8.0, 81)


def _plus():
    return qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())


def _lopsided():
    """sqrt(0.2)|00> + sqrt(0.8)|11>: its entanglement dies in finite time."""
    ket = (np.sqrt(0.2) * qt.basis([2, 2], [0, 0])
           + np.sqrt(0.8) * qt.basis([2, 2], [1, 1]))
    return qt.ket2dm(ket)


def _run(sys_, rho0, op_type="sigma_minus", gamma=0.3, times=TIMES):
    return evolve.master_equation(
        sys_, channels.jump(sys_, op_type, gamma), rho0, times)


def _config(n_qubits, criterion, **kwargs):
    one = n_qubits == 1
    return SystemConfig(
        n_qubits=QubitCount.ONE if one else QubitCount.TWO, omega=1.0,
        J=0.8, interaction_type=InteractionType.XXZ,
        dissipator=DissipatorConfig.constant(0.3), t_max=8.0, dt=0.1,
        decoherence_criterion=criterion, **kwargs)


# --- Negative cases ---


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5, NAN, None])
def test_invalid_threshold_raises_value_error(threshold):
    states = _run(ONE, _plus())
    with pytest.raises(ValueError, match="threshold must be in"):
        target.coherence_crossing(states, ONE, threshold)
    with pytest.raises(ValueError, match="threshold must be in"):
        target.energy_basis_crossing(states, ONE, threshold)


def test_unknown_criterion_raises_value_error():
    with pytest.raises(ValueError, match="criterion must be one of"):
        target.coherence_crossing(_run(ONE, _plus()), ONE, 0.5, "t2")


def test_concurrence_criterion_needs_two_qubits():
    with pytest.raises(ValueError, match="needs two qubits"):
        target.coherence_crossing(
            _run(ONE, _plus()), ONE, 0.5, "concurrence")


def test_sudden_death_needs_two_qubits():
    with pytest.raises(ValueError, match="needs two qubits"):
        target.sudden_death(_run(ONE, _plus()), ONE)


def test_states_of_another_size_than_the_system_raise():
    with pytest.raises(ValueError, match="Hamiltonian has shape"):
        target.coherence_crossing(_run(ONE, _plus()), TWO, 0.5)


def test_empty_states_raise():
    with pytest.raises(ValueError, match="one matrix per time point"):
        target.coherence_crossing(
            evolve.States(np.zeros(0), []), ONE, 0.5)


def test_energy_basis_of_a_non_hermitian_hamiltonian_raises():
    with pytest.raises(ValueError, match="Hermitian"):
        target.energy_basis_coherence(np.array([[0, 1], [0, 0]]), [np.eye(2)])


def test_first_crossing_of_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        target.first_crossing(np.arange(3.0), [1.0, NAN, 0.0], 0.5)


def test_interpolate_crossing_index_out_of_range_raises():
    with pytest.raises(ValueError, match="outside"):
        target.interpolate_crossing(np.arange(3.0), [1.0, 0.5, 0.2], 3, 0.4)


def test_target_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.Target(1.0, False).time = 2.0


# --- Positive cases: equal to the legacy functions ---


def _block_target(config, states, system_):
    """The target block that the criterion of a config selects."""
    kinds = DecoherenceCriterion
    threshold = config.decoherence_threshold
    if config.decoherence_criterion == kinds.COHERENCE_ENERGY:
        return target.energy_basis_crossing(states, system_, threshold)
    if config.decoherence_criterion == kinds.ENTANGLEMENT_DEATH:
        return target.sudden_death(states, system_)
    return target.coherence_crossing(
        states, system_, threshold, config.decoherence_criterion.value)


@pytest.mark.parametrize("criterion", [
    DecoherenceCriterion.COHERENCE, DecoherenceCriterion.PURITY])
def test_one_qubit_targets_equal_compute_t_decoh(criterion):
    config = _config(1, criterion)
    states = _run(ONE, _plus())
    legacy_obs = {
        "coherence_l1": [np.sum(np.abs(r - np.diag(np.diag(r))))
                         for r in states.rho],
        "purity": [np.real(np.trace(r @ r)) for r in states.rho],
    }
    expected = simulator.compute_t_decoh(config, TIMES, legacy_obs)
    got = _block_target(config, states, ONE)
    assert (got.time, got.censored) == expected


@pytest.mark.parametrize("criterion", [
    DecoherenceCriterion.COHERENCE, DecoherenceCriterion.PURITY,
    DecoherenceCriterion.CONCURRENCE, DecoherenceCriterion.COHERENCE_ENERGY,
    DecoherenceCriterion.ENTANGLEMENT_DEATH])
def test_two_qubit_targets_equal_the_legacy_simulation(criterion, monkeypatch):
    config = _config(2, criterion)
    rho0 = _lopsided()
    monkeypatch.setattr(
        initial, "random_pure", lambda system, rng=None: rho0)
    traj = simulator.QuTipSimulator().simulate(config)
    got = _block_target(config, _run(TWO, rho0), TWO)
    assert (got.time, got.censored) == (traj.t_decoh, traj.censored)


def test_energy_basis_crossing_equals_energy_basis_t_decoh():
    config = _config(2, DecoherenceCriterion.COHERENCE_ENERGY)
    states = _run(TWO, _lopsided())
    expected = simulator.energy_basis_t_decoh(
        config, TIMES, TWO.H.full(), states.rho)
    got = target.energy_basis_crossing(
        states, TWO, config.decoherence_threshold)
    assert (got.time, got.censored) == expected


def test_sudden_death_equals_the_legacy_function():
    states = _run(TWO, _lopsided(), gamma=0.5)
    expected = simulator.sudden_death_time(
        TIMES, [measures.wootters_lambda(r) for r in states.rho])
    got = target.sudden_death(states, TWO)
    assert (got.time, got.censored) == expected
    assert not got.censored and 0.0 < got.time < TIMES[-1]


# --- Positive cases: behaviour ---


def test_coherence_crossing_of_exponential_decay_is_the_decay_time():
    # sigma_z dephasing at rate g gives coherence exp(-2 g t) of |+>.
    gamma = 0.3
    states = _run(ONE, _plus(), "sigma_z", gamma,
                  times=np.linspace(0.0, 8.0, 801))
    got = target.coherence_crossing(states, ONE, 1 / math.e)
    assert got.censored is False
    assert got.time == pytest.approx(1 / (2 * gamma), rel=2e-3)


def test_level_that_is_never_reached_is_censored_at_the_last_time():
    states = _run(ONE, _plus(), "sigma_z", 0.001)
    got = target.coherence_crossing(states, ONE, 0.1)
    assert got == target.Target(float(TIMES[-1]), True)


def test_separable_initial_state_has_sudden_death_at_time_zero():
    product = qt.ket2dm(qt.basis([2, 2], [0, 1]))
    got = target.sudden_death(_run(TWO, product), TWO)
    assert got.time == 0.0 and got.censored is False


def test_targets_do_not_need_a_new_simulation():
    states = _run(TWO, _lopsided())
    a = target.coherence_crossing(states, TWO, 0.5, "purity")
    b = target.coherence_crossing(states, TWO, 0.5, "coherence")
    assert a != b
