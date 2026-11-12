"""blocks.evolve: the master-equation solver."""
import numpy as np
import pytest
import qutip as qt

from src.blocks import channels, evolve, system

ONE = system.single_qubit(1.0)
TWO = system.two_qubit(0.5, "XXZ")
TIMES = np.linspace(0.0, 2.0, 5)


def _excited() -> qt.Qobj:
    """The state that sigma_minus empties: qutip's sigmam maps |0> to |1>."""
    return qt.fock_dm(2, 0)


# --- Negative cases ---


@pytest.mark.parametrize("times", [
    np.array([]), np.zeros((2, 2)), np.array([0.0, 1.0, 1.0]),
    np.array([1.0, 0.0]), np.array([0.0, np.nan]), np.array([0.0, np.inf])])
def test_invalid_time_grid_raises_value_error(times):
    with pytest.raises(ValueError, match="times must be"):
        evolve.master_equation(ONE, [], _excited(), times)


def test_initial_state_of_the_wrong_shape_raises_value_error():
    with pytest.raises(ValueError, match="initial state has shape"):
        evolve.master_equation(ONE, [], qt.fock_dm(4, 0), TIMES)


def test_channel_operator_of_the_wrong_shape_raises_value_error():
    bad = [channels.Channel(qt.destroy(4), 0.1)]
    with pytest.raises(ValueError, match="channel operator has shape"):
        evolve.master_equation(ONE, bad, _excited(), TIMES)


# --- Positive cases ---


def test_sigma_minus_decay_matches_the_analytic_solution():
    gamma = 0.3
    chans = channels.jump(ONE, "sigma_minus", gamma)
    states = evolve.master_equation(ONE, chans, _excited(), TIMES)
    populations = np.array([np.real(np.diag(r)) for r in states.rho])
    np.testing.assert_allclose(
        populations[:, 0], np.exp(-gamma * TIMES), atol=1e-6)
    np.testing.assert_allclose(populations.sum(axis=1), 1.0, atol=1e-8)


def test_time_dependent_rate_decays_with_the_integrated_rate():
    chans = channels.bernstein_profile(
        channels.jump(ONE, "sigma_minus", 1.0), [0.2, 0.2], 2.0)
    states = evolve.master_equation(ONE, chans, _excited(), TIMES)
    populations = np.real(np.diag(states.rho[-1]))
    assert populations[0] == pytest.approx(np.exp(-0.4), abs=1e-5)


def test_states_hold_the_grid_and_dense_arrays():
    rho0 = qt.ket2dm(qt.basis([2, 2], [0, 0]))
    states = evolve.master_equation(TWO, [], rho0, TIMES)
    np.testing.assert_array_equal(states.times, TIMES)
    assert len(states.rho) == TIMES.size
    assert all(isinstance(r, np.ndarray) and r.shape == (4, 4)
               for r in states.rho)


def test_no_channels_conserve_purity():
    rho0 = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
    states = evolve.master_equation(ONE, [], rho0, TIMES)
    for r in states.rho:
        assert np.real(np.trace(r @ r)) == pytest.approx(1.0, abs=1e-6)


def test_matches_a_direct_mesolve_call():
    chans = channels.jump(TWO, "sigma_z", 0.2)
    rho0 = qt.ket2dm(qt.basis([2, 2], [1, 0]))
    states = evolve.master_equation(TWO, chans, rho0, TIMES)
    direct = qt.mesolve(
        TWO.H, rho0, TIMES, [np.sqrt(0.2) * c.operator for c in chans],
        e_ops=[], options={"nsteps": 10000}).states
    for ours, theirs in zip(states.rho, direct):
        np.testing.assert_array_equal(ours, theirs.full())
