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


# --- sequential_record ---


def _z_state(bits=(0,)):
    return qt.ket2dm(qt.basis([2] * len(bits), list(bits)))


def _record(sys_, chans, rho0, times, rng_seed=0, **kwargs):
    return evolve.sequential_record(
        sys_, chans, rho0, times, "z", np.random.default_rng(rng_seed),
        **kwargs)


@pytest.mark.parametrize("basis", ["x", "y", "Z", "", None])
def test_sequential_record_unsupported_basis_raises(basis):
    with pytest.raises(ValueError, match="basis must be one of"):
        evolve.sequential_record(
            ONE, [], _excited(), TIMES, basis, np.random.default_rng(0))


@pytest.mark.parametrize("times", [
    np.array([]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0]),
    np.array([0.0, np.nan]), np.zeros((2, 2))])
def test_sequential_record_invalid_times_raise(times):
    with pytest.raises(ValueError, match="times must be"):
        _record(ONE, [], _excited(), times)


def test_sequential_record_state_or_channel_of_the_wrong_shape_raises():
    with pytest.raises(ValueError, match="initial state has shape"):
        _record(ONE, [], qt.fock_dm(4, 0), TIMES)
    with pytest.raises(ValueError, match="channel operator has shape"):
        _record(ONE, [channels.Channel(qt.destroy(4), 0.1)], _excited(),
                TIMES)


def test_sequential_record_without_a_generator_raises_type_error():
    with pytest.raises(TypeError, match="Generator"):
        evolve.sequential_record(ONE, [], _excited(), TIMES, "z", 5)


def test_zeno_outcomes_stay_constant_after_the_first_measurement():
    plus = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
    for seed in range(20):
        record = _record(ONE, [], plus, TIMES, rng_seed=seed)
        assert record.outcomes.shape == (len(TIMES), 1)
        assert np.all(record.outcomes == record.outcomes[0])


def test_zeno_holds_for_two_qubits_with_a_z_diagonal_hamiltonian():
    diagonal = system.System(
        H=qt.tensor(qt.sigmaz(), qt.sigmaz()),
        spin_ops=TWO.spin_ops, positions=TWO.positions, n_qubits=2)
    state = qt.ket2dm(
        (qt.basis([2, 2], [0, 1]) + qt.basis([2, 2], [1, 0])).unit())
    for seed in range(10):
        record = _record(diagonal, [], state, TIMES, rng_seed=seed)
        assert np.all(record.outcomes == record.outcomes[0])
        assert record.outcomes[0, 0] != record.outcomes[0, 1]


def test_amplitude_damping_populations_follow_the_exponential():
    gamma, p0 = 0.4, 0.8
    rho0 = qt.Qobj(np.diag([p0, 1 - p0]), dims=[[2], [2]])
    chans = channels.jump(ONE, "sigma_minus", gamma)
    times = np.array([0.0, 0.5, 1.0, 2.0])
    n = 1500
    outcomes = np.array([
        _record(ONE, chans, rho0, times, rng_seed=seed).outcomes[:, 0]
        for seed in range(n)])
    # Outcome 0 is sigma_z = +1, the level that sigma_minus empties.
    fraction = (outcomes == 0).mean(axis=0)
    # Decay is irreversible, so the collapse at every measurement does not
    # change the probability of finding the level still excited.
    expected = p0 * np.exp(-gamma * times)
    sigma = np.sqrt(expected * (1 - expected) / n)
    assert np.all(np.abs(fraction - expected) < 5 * sigma)


def test_the_same_generator_seed_gives_the_same_record():
    plus = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
    chans = channels.jump(ONE, "sigma_minus", 0.5)
    a = _record(ONE, chans, plus, TIMES, rng_seed=4, keep_rho=True)
    b = _record(ONE, chans, plus, TIMES, rng_seed=4, keep_rho=True)
    c = _record(ONE, chans, plus, TIMES, rng_seed=5)
    np.testing.assert_array_equal(a.outcomes, b.outcomes)
    for x, y in zip(a.rho_after, b.rho_after):
        np.testing.assert_array_equal(x, y)
    assert c.rho_after is None
    outcomes = np.array([
        _record(ONE, chans, plus, TIMES, rng_seed=s).outcomes[:, 0]
        for s in range(10)])
    assert len({tuple(row) for row in outcomes}) > 1


def test_post_measurement_states_are_the_basis_states_of_the_outcomes():
    plus = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
    record = _record(ONE, [], plus, TIMES, keep_rho=True)
    for outcome, rho in zip(record.outcomes[:, 0], record.rho_after):
        expected = np.zeros((2, 2))
        expected[outcome, outcome] = 1.0
        np.testing.assert_allclose(rho, expected, atol=1e-12)


def test_the_record_keeps_the_time_grid_and_integer_outcomes():
    record = _record(TWO, [], _z_state((0, 1)), TIMES)
    np.testing.assert_array_equal(record.times, TIMES)
    assert record.outcomes.dtype.kind == "i"
    assert record.outcomes.shape == (len(TIMES), 2)
