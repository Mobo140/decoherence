"""simulate_qutip: the master-equation solver and its input checks.

Every simulation here has a handful of time points.
"""
import numpy as np
import pytest
import qutip as qt

from src.physics import lindblad_systems, simulate_qutip

TIMES = np.linspace(0.0, 2.0, 5)
NAN = float("nan")


def _plus() -> qt.Qobj:
    return qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())


def _two_qubit_state() -> qt.Qobj:
    return qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 1)))


# --- collapse operators ----------------------------------------------------

def test_constant_rate_collapse_operators_are_scaled_by_sqrt_gamma():
    system = lindblad_systems.SingleQubitSystem(gamma_const=0.4)
    (op,) = simulate_qutip._collapse_operators(system)
    np.testing.assert_allclose(op.full(), np.sqrt(0.4) * qt.sigmam().full())


def test_time_dependent_collapse_operators_pair_with_sqrt_of_the_rate():
    system = lindblad_systems.SingleQubitSystem(
        gamma_coeffs=np.array([0.0, 0.4]), gamma_t_max=2.0)
    ((op, coefficient),) = simulate_qutip._collapse_operators(system)
    np.testing.assert_allclose(op.full(), qt.sigmam().full())
    assert coefficient(0.0, {}) == 0.0
    assert coefficient(2.0, {}) == pytest.approx(np.sqrt(0.4))
    assert coefficient(1.0, {}) == pytest.approx(np.sqrt(0.2))


# --- one qubit -------------------------------------------------------------

def test_single_qubit_returns_the_default_observables_and_density_matrices():
    system = lindblad_systems.SingleQubitSystem(gamma_const=0.4)
    out = simulate_qutip.simulate_single_qubit(system, _plus(), TIMES)
    assert set(out) == {
        "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1",
        "density_matrices"}
    assert len(out["density_matrices"]) == len(TIMES)
    for name in ("sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"):
        assert out[name].shape == TIMES.shape
    assert out["density_matrices"][0].shape == (2, 2)


def test_single_qubit_coherence_decays_as_exp_minus_gamma_t_over_two():
    gamma = 0.6
    system = lindblad_systems.SingleQubitSystem(gamma_const=gamma)
    out = simulate_qutip.simulate_single_qubit(system, _plus(), TIMES)
    np.testing.assert_allclose(
        out["coherence_l1"], np.exp(-gamma * TIMES / 2), atol=1e-5)
    assert out["purity"][0] == pytest.approx(1.0)
    assert out["purity"][-1] < out["purity"][0]


def test_single_qubit_can_record_a_subset_of_the_pauli_expectations():
    system = lindblad_systems.SingleQubitSystem(gamma_const=0.1)
    out = simulate_qutip.simulate_single_qubit(
        system, _plus(), TIMES, observables=["sigma_x"])
    assert set(out) == {
        "sigma_x", "purity", "coherence_l1", "density_matrices"}


def test_single_qubit_without_dissipation_keeps_the_state_pure():
    system = lindblad_systems.SingleQubitSystem(omega=1.0)
    out = simulate_qutip.simulate_single_qubit(system, _plus(), TIMES)
    np.testing.assert_allclose(out["purity"], 1.0, atol=1e-6)


def test_single_qubit_time_dependent_constant_coefficients_match_a_constant():
    constant = lindblad_systems.SingleQubitSystem(gamma_const=0.3)
    polynomial = lindblad_systems.SingleQubitSystem(
        gamma_coeffs=np.array([0.3, 0.3]), gamma_t_max=2.0)
    a = simulate_qutip.simulate_single_qubit(constant, _plus(), TIMES)
    b = simulate_qutip.simulate_single_qubit(polynomial, _plus(), TIMES)
    np.testing.assert_allclose(
        a["coherence_l1"], b["coherence_l1"], atol=1e-5)


def test_single_qubit_zero_rate_polynomial_does_not_dissipate():
    system = lindblad_systems.SingleQubitSystem(
        gamma_coeffs=np.array([0.0, 0.0]), gamma_t_max=2.0)
    out = simulate_qutip.simulate_single_qubit(system, _plus(), TIMES)
    np.testing.assert_allclose(out["purity"], 1.0, atol=1e-6)


def test_single_qubit_accepts_a_single_time_point():
    system = lindblad_systems.SingleQubitSystem(gamma_const=0.1)
    out = simulate_qutip.simulate_single_qubit(
        system, _plus(), np.array([0.0]))
    assert out["purity"].shape == (1,)


# --- two qubits ------------------------------------------------------------

def test_two_qubit_returns_the_default_observables():
    system = lindblad_systems.TwoQubitSystem(gamma_const=0.2)
    out = simulate_qutip.simulate_two_qubit(
        system, _two_qubit_state(), TIMES)
    assert set(out) == {
        "sigma_x1", "sigma_y1", "sigma_z1", "sigma_x2", "sigma_y2",
        "sigma_z2", "purity", "coherence_l1", "density_matrices"}
    assert out["density_matrices"][0].shape == (4, 4)
    assert out["sigma_z1"][0] == pytest.approx(1.0)
    assert out["sigma_z2"][0] == pytest.approx(-1.0)
    assert np.all(out["purity"] <= 1.0 + 1e-9)


def test_two_qubit_can_record_a_subset_of_the_pauli_expectations():
    system = lindblad_systems.TwoQubitSystem(gamma_const=0.2)
    out = simulate_qutip.simulate_two_qubit(
        system, _two_qubit_state(), TIMES,
        observables=["sigma_x1", "sigma_z2"])
    assert set(out) == {
        "sigma_x1", "sigma_z2", "purity", "coherence_l1", "density_matrices"}


def test_two_qubit_time_dependent_rate_conserves_the_trace():
    system = lindblad_systems.TwoQubitSystem(
        gamma_coeffs=np.array([0.1, 0.5]), gamma_t_max=2.0)
    out = simulate_qutip.simulate_two_qubit(
        system, _two_qubit_state(), TIMES)
    traces = [np.trace(rho).real for rho in out["density_matrices"]]
    np.testing.assert_allclose(traces, 1.0, atol=1e-6)


# --- Negative cases ---


def _system() -> lindblad_systems.SingleQubitSystem:
    return lindblad_systems.SingleQubitSystem(gamma_const=0.1)


def test_simulate_single_qubit_empty_times_raise_value_error():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        simulate_qutip.simulate_single_qubit(_system(), _plus(), np.array([]))


def test_simulate_single_qubit_two_dimensional_times_raise_value_error():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        simulate_qutip.simulate_single_qubit(
            _system(), _plus(), np.zeros((2, 2)))


@pytest.mark.parametrize("times", [
    [1.0, 0.5], [0.0, 1.0, 1.0], [0.0, NAN, 2.0], [0.0, float("inf")]])
def test_simulate_single_qubit_unordered_or_non_finite_times_raise(times):
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        simulate_qutip.simulate_single_qubit(
            _system(), _plus(), np.array(times))


def test_simulate_single_qubit_two_qubit_state_raises_value_error():
    with pytest.raises(ValueError, match="initial state has shape"):
        simulate_qutip.simulate_single_qubit(
            _system(), _two_qubit_state(), TIMES)


def test_simulate_two_qubit_one_qubit_state_raises_value_error():
    system = lindblad_systems.TwoQubitSystem(gamma_const=0.1)
    with pytest.raises(ValueError, match="initial state has shape"):
        simulate_qutip.simulate_two_qubit(system, _plus(), TIMES)


def test_simulate_single_qubit_unknown_observable_lists_the_valid_ones():
    with pytest.raises(ValueError, match="sigma_q.*sigma_x"):
        simulate_qutip.simulate_single_qubit(
            _system(), _plus(), TIMES, observables=["sigma_q"])


def test_simulate_single_qubit_two_qubit_observable_name_raises():
    with pytest.raises(ValueError, match="Unknown observables"):
        simulate_qutip.simulate_single_qubit(
            _system(), _plus(), TIMES, observables=["sigma_x1"])


def test_simulate_two_qubit_one_qubit_observable_name_raises():
    system = lindblad_systems.TwoQubitSystem(gamma_const=0.1)
    with pytest.raises(ValueError, match="Unknown observables"):
        simulate_qutip.simulate_two_qubit(
            system, _two_qubit_state(), TIMES, observables=["sigma_x"])
