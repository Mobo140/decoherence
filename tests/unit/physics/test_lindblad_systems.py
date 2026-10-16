"""lindblad_systems: Bernstein rates, the Lindblad systems, random states."""
import math

import numpy as np
import pytest
import qutip as qt

from src.physics import lindblad_systems
from src.physics.lindblad_systems import (
    bernstein_polynomial,
    bernstein_gamma,
    SingleQubitSystem,
    TwoQubitSystem,
    sample_random_pure_state,
)

NAN = float("nan")
INF = float("inf")


def _kron(a: qt.Qobj, b: qt.Qobj) -> np.ndarray:
    return qt.tensor(a, b).full()


# --- Bernstein rate --------------------------------------------------------

def test_bernstein_polynomial_out_of_range_index_is_zero_and_basis_sums_to_one():
    # k out of range
    assert bernstein_polynomial(3, -1, 0.5) == 0.0
    assert bernstein_polynomial(3, 4, 0.5) == 0.0

    # sum of Bernstein basis at t should be 1
    vals = [bernstein_polynomial(4, k, 0.3) for k in range(5)]
    assert np.isclose(sum(vals), 1.0)


def test_bernstein_polynomial_known_values():
    assert bernstein_polynomial(2, 1, 0.5) == pytest.approx(0.5)
    assert bernstein_polynomial(2, 0, 0.5) == pytest.approx(0.25)
    assert bernstein_polynomial(3, 0, 0.0) == 1.0
    assert bernstein_polynomial(3, 3, 1.0) == 1.0
    assert bernstein_polynomial(3, 1, 0.0) == 0.0


def test_bernstein_gamma_is_non_negative():
    coeffs = np.array([0.1, 0.2, 0.3])
    for t in [0.0, 0.5, 1.0]:
        g = bernstein_gamma(t, coeffs, t_max=2.0)
        assert g >= 0.0


def test_bernstein_gamma_interpolates_between_the_end_coefficients():
    coeffs = np.array([0.2, 0.6])
    assert bernstein_gamma(0.0, coeffs, t_max=4.0) == pytest.approx(0.2)
    assert bernstein_gamma(2.0, coeffs, t_max=4.0) == pytest.approx(0.4)
    assert bernstein_gamma(4.0, coeffs, t_max=4.0) == pytest.approx(0.6)


def test_bernstein_gamma_clamps_times_outside_the_window():
    coeffs = np.array([0.2, 0.6])
    assert bernstein_gamma(-3.0, coeffs, t_max=4.0) == pytest.approx(0.2)
    assert bernstein_gamma(40.0, coeffs, t_max=4.0) == pytest.approx(0.6)


def test_bernstein_gamma_clips_a_negative_polynomial_to_zero():
    assert bernstein_gamma(0.5, np.array([-0.2, -0.4]), t_max=1.0) == 0.0


def test_bernstein_gamma_single_coefficient_is_a_constant_rate():
    for t in (0.0, 0.7, 3.0):
        assert bernstein_gamma(t, np.array([0.3]), t_max=3.0) == 0.3


# --- SingleQubitSystem -----------------------------------------------------

def test_single_qubit_system_gives_gamma_and_lindblad_operators():
    sys = SingleQubitSystem(gamma_const=0.2)
    assert np.isclose(sys.get_gamma(0.0), 0.2)
    ops = sys.get_lindblad_operators(0.0)
    assert len(ops) > 0 and isinstance(ops[0][0], object)


def test_single_qubit_system_hamiltonian_is_omega_sigma_z():
    system = SingleQubitSystem(omega=1.7, gamma_const=0.1)
    np.testing.assert_allclose(system.H.full(), 1.7 * qt.sigmaz().full())


def test_single_qubit_system_defaults_to_sigma_minus_jumps():
    system = SingleQubitSystem(gamma_const=0.1)
    assert system.jump_operators == ["sigma_minus"]
    np.testing.assert_allclose(system.L_list[0].full(), qt.sigmam().full())


def test_single_qubit_system_sigma_z_and_several_jump_operators():
    system = SingleQubitSystem(
        gamma_const=0.1, jump_operators=["sigma_z", "sigma_minus"])
    assert len(system.L_list) == 2
    np.testing.assert_allclose(system.L_list[0].full(), qt.sigmaz().full())
    assert [rate for _, rate in system.get_lindblad_operators(0.0)] == [
        0.1, 0.1]


def test_single_qubit_system_time_dependent_rate_follows_the_polynomial():
    coeffs = np.array([0.1, 0.3])
    system = SingleQubitSystem(gamma_coeffs=coeffs, gamma_t_max=2.0)
    assert system.get_gamma(1.0) == pytest.approx(0.2)
    assert system.get_lindblad_operators(2.0)[0][1] == pytest.approx(0.3)


def test_single_qubit_system_constant_rate_wins_over_coefficients():
    system = SingleQubitSystem(
        gamma_const=0.5, gamma_coeffs=np.array([0.1, 0.1]))
    assert system.get_gamma(0.3) == 0.5


def test_single_qubit_system_without_a_rate_does_not_dissipate():
    assert SingleQubitSystem().get_gamma(1.0) == 0.0


# --- TwoQubitSystem --------------------------------------------------------

def test_two_qubit_xxz_hamiltonian_is_the_isotropic_heisenberg_coupling():
    system = TwoQubitSystem(J=0.8, interaction_type="XXZ", gamma_const=0.1)
    expected = 0.8 * sum(
        _kron(p, p) for p in (qt.sigmax(), qt.sigmay(), qt.sigmaz()))
    np.testing.assert_allclose(system.H.full(), expected)


def test_two_qubit_tfim_hamiltonian_has_unit_transverse_field():
    system = TwoQubitSystem(J=0.8, interaction_type="TFIM", gamma_const=0.1)
    eye = qt.qeye(2)
    expected = (0.8 * _kron(qt.sigmax(), qt.sigmax())
                + _kron(qt.sigmaz(), eye) + _kron(eye, qt.sigmaz()))
    np.testing.assert_allclose(system.H.full(), expected)
    assert system.H.isherm


def test_two_qubit_system_has_one_jump_operator_per_qubit_and_type():
    system = TwoQubitSystem(gamma_const=0.1)
    assert len(system.L_list) == 2
    assert all(op.shape == (4, 4) for op in system.L_list)
    both = TwoQubitSystem(
        gamma_const=0.1, jump_operators=["sigma_minus", "sigma_z"])
    assert len(both.L_list) == 4
    np.testing.assert_allclose(
        both.L_list[2].full(), _kron(qt.sigmaz(), qt.qeye(2)))
    np.testing.assert_allclose(
        both.L_list[3].full(), _kron(qt.qeye(2), qt.sigmaz()))


def test_two_qubit_system_uses_destroy_for_sigma_minus():
    system = TwoQubitSystem(gamma_const=0.1)
    np.testing.assert_allclose(
        system.L_list[0].full(), _kron(qt.destroy(2), qt.qeye(2)))
    np.testing.assert_allclose(
        system.L_list[1].full(), _kron(qt.qeye(2), qt.destroy(2)))


def test_two_qubit_system_shares_one_rate_between_the_jump_operators():
    system = TwoQubitSystem(
        gamma_coeffs=np.array([0.1, 0.3]), gamma_t_max=2.0)
    rates = [rate for _, rate in system.get_lindblad_operators(1.0)]
    assert rates == [pytest.approx(0.2)] * 2


def test_two_qubit_system_without_a_rate_does_not_dissipate():
    # The J sweeps of R9 build the Hamiltonian only.
    system = TwoQubitSystem(J=0.4, interaction_type="TFIM")
    assert system.get_gamma(1.0) == 0.0 and system.H.shape == (4, 4)


# --- random pure states ----------------------------------------------------

def test_sample_random_pure_state_same_seed_gives_same_state():
    s1 = sample_random_pure_state(n_qubits=1, seed=123)
    s2 = sample_random_pure_state(n_qubits=1, seed=123)
    assert np.allclose(s1.full(), s2.full())


def test_sample_random_pure_state_is_a_normalised_pure_density_matrix():
    for n_qubits in (1, 2):
        rho = sample_random_pure_state(n_qubits=n_qubits, seed=5)
        dim = 2 ** n_qubits
        assert rho.dims == [[2] * n_qubits, [2] * n_qubits]
        assert rho.shape == (dim, dim)
        assert rho.tr() == pytest.approx(1.0)
        assert (rho * rho).tr().real == pytest.approx(1.0)
        assert rho.isherm


def test_sample_random_pure_state_different_seeds_differ():
    a = sample_random_pure_state(1, seed=1).full()
    b = sample_random_pure_state(1, seed=2).full()
    assert not np.allclose(a, b)


def test_sample_random_pure_state_without_a_seed_follows_the_global_stream():
    np.random.seed(7)
    first = sample_random_pure_state(2).full()
    np.random.seed(7)
    again = sample_random_pure_state(2).full()
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, sample_random_pure_state(2).full())


def test_sample_random_pure_state_seed_does_not_touch_the_global_stream():
    np.random.seed(3)
    expected = np.random.rand()
    np.random.seed(3)
    sample_random_pure_state(1, seed=99)
    assert np.random.rand() == expected


# --- Negative cases ---


@pytest.mark.parametrize("value", [-0.1, NAN, INF])
def test_single_qubit_system_invalid_constant_rate_raises_value_error(value):
    with pytest.raises(ValueError, match="gamma_const"):
        SingleQubitSystem(gamma_const=value)


@pytest.mark.parametrize("value", [-0.1, NAN, INF])
def test_two_qubit_system_invalid_constant_rate_raises_value_error(value):
    with pytest.raises(ValueError, match="gamma_const"):
        TwoQubitSystem(gamma_const=value)


def test_systems_empty_coefficients_raise_value_error():
    with pytest.raises(ValueError, match="gamma_coeffs must not be empty"):
        SingleQubitSystem(gamma_coeffs=np.array([]))
    with pytest.raises(ValueError, match="gamma_coeffs must not be empty"):
        TwoQubitSystem(gamma_coeffs=np.array([]))


@pytest.mark.parametrize("t_max", [0.0, -1.0, NAN, INF])
def test_systems_invalid_gamma_t_max_raise_value_error(t_max):
    with pytest.raises(ValueError, match="gamma_t_max"):
        SingleQubitSystem(gamma_coeffs=np.array([0.1]), gamma_t_max=t_max)
    with pytest.raises(ValueError, match="gamma_t_max"):
        TwoQubitSystem(gamma_coeffs=np.array([0.1]), gamma_t_max=t_max)


def test_single_qubit_system_unknown_jump_operator_raises_value_error():
    with pytest.raises(ValueError, match="Unknown jump operator: sigma_x"):
        SingleQubitSystem(gamma_const=0.1, jump_operators=["sigma_x"])


def test_two_qubit_system_unknown_jump_operator_raises_value_error():
    with pytest.raises(ValueError, match="Unknown jump operator type: foo"):
        TwoQubitSystem(gamma_const=0.1, jump_operators=["foo"])


@pytest.mark.parametrize("interaction", ["XYZ", "xxz", "", None])
def test_two_qubit_system_unknown_interaction_raises_value_error(interaction):
    with pytest.raises(ValueError, match="Unknown interaction type"):
        TwoQubitSystem(interaction_type=interaction, gamma_const=0.1)


@pytest.mark.parametrize("t_max", [0.0, -2.0, NAN])
def test_bernstein_gamma_non_positive_t_max_raises_value_error(t_max):
    with pytest.raises(ValueError, match="t_max must be positive"):
        bernstein_gamma(0.5, np.array([0.1, 0.2]), t_max)


def test_bernstein_gamma_empty_coefficients_raise_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        bernstein_gamma(0.5, np.array([]), 1.0)


@pytest.mark.parametrize("n_qubits", [0, -1, 1.5, 0.5])
def test_sample_random_pure_state_invalid_qubit_count_raises_value_error(
        n_qubits):
    with pytest.raises(ValueError, match="positive integer"):
        sample_random_pure_state(n_qubits)


def test_bernstein_polynomial_out_of_range_degree_or_index_gives_zero():
    # Documented: an index outside 0..degree is not an error.
    assert bernstein_polynomial(-1, 0, 0.5) == 0.0
    assert bernstein_polynomial(2, 5, 0.5) == 0.0


def test_bernstein_polynomial_does_not_check_the_argument_range():
    # Debatable, kept: t outside [0, 1] extrapolates (hot path of the ODE).
    assert bernstein_polynomial(3, 1, 1.5) == pytest.approx(
        3 * 1.5 * 0.5 ** 2)
    assert math.isnan(bernstein_polynomial(3, 1, NAN))


def test_single_qubit_system_accepts_degenerate_but_physical_settings():
    # Debatable, kept: a negative omega is a valid Hamiltonian, an empty
    # jump list is a closed system.
    assert SingleQubitSystem(omega=-1.0, gamma_const=0.1).H.full()[0, 0] == -1
    assert SingleQubitSystem(gamma_const=0.1, jump_operators=[]).L_list == []
