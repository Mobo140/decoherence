"""kondo_bath: kernels, Korringa rates, Davies collapse operators, shifts."""
import math

import numpy as np
import pytest
import qutip as qt
from scipy import special

from src.physics import kondo_bath

RHO_J = 0.1
KT = 1.0
PREF = math.pi * RHO_J ** 2 / 4.0


def _gamma0(delta: float, kt: float = KT) -> float:
    """Single spin-1/2 Zeeman relaxation rate, PRB Eq. 22."""
    return math.pi * RHO_J ** 2 / 8.0 * float(kondo_bath.bath_rate(delta, kt))


def _kwargs(dim=3, k_f=1.0, kt=KT, rho_j=RHO_J):
    return dict(rho_J=rho_j, kT=kt, k_F=k_f, dim=dim)


def _dimer_hamiltonian(j_h: float, field: float = 0.0) -> qt.Qobj:
    ops = kondo_bath.chain_spin_operators(2)
    heis = sum(ops[0][a] * ops[1][a] for a in range(3))
    return j_h * heis + field * (ops[0][2] + ops[1][2])


def _random_hermitian(n: int, seed: int) -> qt.Qobj:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return qt.Qobj((a + a.conj().T) / 2, dims=[[2] * (n // 2)] * 2)


def _dissipator(c_ops, rho: qt.Qobj) -> qt.Qobj:
    return sum(qt.lindblad_dissipator(c) * qt.operator_to_vector(rho)
               for c in c_ops)


# --- kernels and the bath rate function ------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pair_kernel_is_one_at_zero_distance_and_bounded(dim):
    x = np.linspace(0.0, 30.0, 301)
    values = kondo_bath.pair_kernel(dim, x)
    assert values[0] == pytest.approx(1.0)
    assert np.all(values >= 0.0) and np.all(values <= 1.0 + 1e-12)


def test_pair_kernel_matches_closed_forms():
    x = np.array([0.3, 1.0, 2.5, 7.0])
    np.testing.assert_allclose(kondo_bath.pair_kernel(1, x), np.cos(x) ** 2)
    np.testing.assert_allclose(
        kondo_bath.pair_kernel(2, x), special.j0(x) ** 2)
    np.testing.assert_allclose(
        kondo_bath.pair_kernel(3, x), (np.sin(x) / x) ** 2)


def test_bath_rate_elastic_limit_is_kT():
    assert kondo_bath.bath_rate(0.0, 0.7) == pytest.approx(0.7)
    assert kondo_bath.bath_rate(1e-12, 0.7) == pytest.approx(0.7)


def test_bath_rate_large_gap_is_linear_in_the_gap():
    assert kondo_bath.bath_rate(50.0, 1.0) == pytest.approx(50.0)
    assert kondo_bath.bath_rate(-50.0, 1.0) == pytest.approx(
        50.0 * math.exp(-50.0), rel=1e-9)


def test_bath_rate_satisfies_detailed_balance_and_accepts_arrays():
    delta = np.array([0.1, 1.0, 3.0, 20.0])
    up = kondo_bath.bath_rate(-delta, 0.8)
    down = kondo_bath.bath_rate(delta, 0.8)
    np.testing.assert_allclose(up, down * np.exp(-delta / 0.8), rtol=1e-10)
    assert down.shape == delta.shape


def test_chain_spin_operators_obey_the_algebra_and_sites_commute():
    for spin in (0.5, 1.0, 1.5):
        ops = kondo_bath.chain_spin_operators(2, spin)
        sx, sy, sz = ops[0]
        assert (sx * sy - sy * sx - 1j * sz).norm() < 1e-12
        total = sx * sx + sy * sy + sz * sz
        dim = int(2 * spin + 1)
        expected = spin * (spin + 1) * np.eye(dim ** 2)
        np.testing.assert_allclose(total.full(), expected, atol=1e-12)
        assert (ops[0][0] * ops[1][1] - ops[1][1] * ops[0][0]).norm() < 1e-12


def test_kernel_matrix_is_symmetric_with_unit_diagonal():
    positions = np.array([0.0, 0.7, 1.9])
    kernel = kondo_bath.kernel_matrix(positions, k_F=1.3, dim=2)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    assert kernel[0, 1] == pytest.approx(special.j0(1.3 * 0.7) ** 2)


def test_kernel_matrix_accepts_points_in_the_plane():
    points = np.array([[0.0, 0.0], [3.0, 4.0]])
    kernel = kondo_bath.kernel_matrix(points, k_F=1.0, dim=1)
    assert kernel[0, 1] == pytest.approx(math.cos(5.0) ** 2)


# --- rates ------------------------------------------------------------------

def test_single_spin_zeeman_rate_equals_gamma0():
    delta = 3.0
    ops = kondo_bath.chain_spin_operators(1)
    ham = delta / 2 * qt.sigmaz()
    rates = kondo_bath.kondo_rates(ham, ops, [0.0], **_kwargs())
    # Eigenvalues are ascending: index 0 is the ground state (down).
    assert rates.rates[1, 0] == pytest.approx(_gamma0(delta))
    assert rates.rates[0, 1] == pytest.approx(_gamma0(-delta))
    assert rates.rates[0, 0] == 0.0


def test_single_spin_rate_is_gamma0_for_elastic_limit():
    ops = kondo_bath.chain_spin_operators(1)
    rates = kondo_bath.kondo_rates(0.0 * qt.sigmaz(), ops, [0.0], **_kwargs())
    assert rates.rates[0, 1] == pytest.approx(_gamma0(0.0))


def test_single_spin_adiabatic_dephasing_matches_eq_39_and_scales_as_s2():
    gammas = {}
    for spin in (0.5, 1.0, 1.5, 2.5):
        ops = kondo_bath.chain_spin_operators(1, spin)
        dim = int(2 * spin + 1)
        states = np.eye(dim)
        ham = 0.0 * ops[0][2]
        rates = kondo_bath.kondo_rates(
            ham, ops, [0.0], states=states, **_kwargs())
        # Between m = +S and m = -S: Eq. 39, pi (rho J)^2 / 2 * kT * S^2.
        gammas[spin] = rates.dephasing_adiabatic[0, dim - 1]
        assert gammas[spin] == pytest.approx(
            math.pi * RHO_J ** 2 / 2 * KT * spin ** 2)
    assert gammas[2.5] / gammas[0.5] == pytest.approx(25.0)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_dimer_singlet_triplet_rate_follows_one_minus_kernel(dim):
    j_h = 5.0
    ham = _dimer_hamiltonian(j_h)
    for k_f_d in (0.0, 0.4, 1.0, 2.3, 5.0):
        rates = kondo_bath.kondo_rates(
            ham, kondo_bath.chain_spin_operators(2), [0.0, k_f_d],
            **_kwargs(dim=dim))
        kernel = float(kondo_bath.pair_kernel(dim, k_f_d))
        # Index 0 is the singlet, 1-3 the triplet.
        for triplet in (1, 2, 3):
            assert rates.rates[triplet, 0] == pytest.approx(
                _gamma0(j_h) * (1.0 - kernel), abs=1e-12)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_dimer_triplet_to_triplet_rate_in_a_large_field_follows_one_plus_kernel(
        dim):
    j_h, field = 5.0, 10.0
    ham = _dimer_hamiltonian(j_h, field)
    energies, vectors = ham.eigenstates()
    # |1,-1> = |down down> is the ground state; |1,0> is the next one up
    # among the states that couple to it.
    down_down = qt.tensor(qt.basis(2, 1), qt.basis(2, 1)).full().ravel()
    triplet_zero = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
                    + qt.tensor(qt.basis(2, 1), qt.basis(2, 0))).full().ravel()
    triplet_zero /= np.linalg.norm(triplet_zero)
    for k_f_d in (0.0, 0.8, 2.0, 4.0):
        rates = kondo_bath.kondo_rates(
            ham, kondo_bath.chain_spin_operators(2), [0.0, k_f_d],
            **_kwargs(dim=dim))
        overlap = np.abs(rates.states.conj().T @ triplet_zero)
        i_zero = int(np.argmax(overlap))
        overlap = np.abs(rates.states.conj().T @ down_down)
        i_ground = int(np.argmax(overlap))
        kernel = float(kondo_bath.pair_kernel(dim, k_f_d))
        assert rates.rates[i_zero, i_ground] == pytest.approx(
            _gamma0(field) * (1.0 + kernel), rel=1e-9)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_ferromagnetic_ising_dephasing_grows_as_n_squared_when_k_f_d_vanishes(
        dim):
    gammas = {}
    for n_sites in (1, 2, 3, 4):
        ops = kondo_bath.chain_spin_operators(n_sites)
        ham = -sum(ops[i][2] * ops[i + 1][2] for i in range(n_sites - 1))
        if n_sites == 1:
            ham = 0.0 * ops[0][2]
        positions = 1e-6 * np.arange(n_sites)
        rates = kondo_bath.kondo_rates(
            ham, ops, positions, states=np.eye(2 ** n_sites),
            **_kwargs(dim=dim))
        gammas[n_sites] = rates.dephasing_adiabatic[0, 1]
    for n_sites in (2, 3, 4):
        assert gammas[n_sites] / gammas[1] == pytest.approx(
            n_sites ** 2, rel=1e-6)


def test_rates_obey_detailed_balance_for_a_generic_hamiltonian():
    ops = kondo_bath.chain_spin_operators(2)
    ham = _random_hermitian(4, 3)
    rates = kondo_bath.kondo_rates(ham, ops, [0.0, 1.1], **_kwargs(kt=0.9))
    for m in range(4):
        for k in range(4):
            if m == k:
                continue
            ratio = rates.rates[m, k] / rates.rates[k, m]
            expected = math.exp(-(rates.energies[k] - rates.energies[m]) / 0.9)
            assert ratio == pytest.approx(expected, rel=1e-8)


def test_nonadiabatic_dephasing_is_half_the_sum_of_the_leaving_rates():
    ops = kondo_bath.chain_spin_operators(2)
    ham = _random_hermitian(4, 5)
    rates = kondo_bath.kondo_rates(ham, ops, [0.0, 1.1], **_kwargs())
    out = rates.rates.sum(axis=1)
    total = rates.dephasing - rates.dephasing_adiabatic
    np.testing.assert_allclose(
        total, 0.5 * (out[:, None] + out[None, :]), atol=1e-12)


# --- collapse operators ----------------------------------------------------

def test_collapse_operators_reproduce_the_pauli_master_equation():
    ops = kondo_bath.chain_spin_operators(2)
    ham = _random_hermitian(4, 11)
    kw = _kwargs(kt=0.8)
    rates = kondo_bath.kondo_rates(ham, ops, [0.0, 1.3], **kw)
    c_ops = kondo_bath.kondo_collapse_operators(ham, ops, [0.0, 1.3], **kw)
    populations = np.array([0.1, 0.4, 0.3, 0.2])
    rho = qt.Qobj(
        rates.states @ np.diag(populations) @ rates.states.conj().T,
        dims=ham.dims)
    drho = qt.vector_to_operator(_dissipator(c_ops, rho)).full()
    dp = np.real(np.diag(rates.states.conj().T @ drho @ rates.states))
    expected = rates.rates.T @ populations - rates.rates.sum(1) * populations
    np.testing.assert_allclose(dp, expected, atol=1e-12)


@pytest.mark.parametrize("ham_name", ["generic", "dimer", "ising"])
def test_dissipator_is_secular_also_for_degenerate_spectra(ham_name):
    ops = kondo_bath.chain_spin_operators(2)
    ham = {
        "generic": _random_hermitian(4, 2),
        "dimer": _dimer_hamiltonian(1.0),
        "ising": -ops[0][2] * ops[1][2],
    }[ham_name]
    c_ops = kondo_bath.kondo_collapse_operators(
        ham, ops, [0.0, 1.0], **_kwargs())
    rho = qt.rand_dm([2, 2], seed=7)
    commutator = ham * rho - rho * ham
    lhs = _dissipator(c_ops, commutator)
    rhs = (qt.operator_to_vector(ham * qt.vector_to_operator(
        _dissipator(c_ops, rho)) - qt.vector_to_operator(
            _dissipator(c_ops, rho)) * ham))
    assert (lhs - rhs).norm() < 1e-12


@pytest.mark.parametrize("ham_name", ["generic", "dimer", "ising_field"])
def test_steady_state_is_the_gibbs_state(ham_name):
    ops = kondo_bath.chain_spin_operators(2)
    ham = {
        "generic": _random_hermitian(4, 9),
        "dimer": _dimer_hamiltonian(0.7),
        "ising_field": -ops[0][2] * ops[1][2] + 0.3 * (
            ops[0][0] + ops[1][0]),
    }[ham_name]
    kt = 0.8
    c_ops = kondo_bath.kondo_collapse_operators(
        ham, ops, [0.0, 1.0], **_kwargs(kt=kt))
    steady = qt.steadystate(ham, c_ops)
    gibbs = (-ham / kt).expm()
    gibbs = gibbs / gibbs.tr()
    assert (steady - gibbs).norm() < 1e-8


def test_nearly_degenerate_levels_within_tol_are_grouped_as_degenerate():
    ops = kondo_bath.chain_spin_operators(1)
    exact = 0.0 * qt.sigmaz()
    split = 1e-12 * qt.sigmaz()
    ref = kondo_bath.kondo_collapse_operators(
        exact, ops, [0.0], **_kwargs())
    near = kondo_bath.kondo_collapse_operators(
        split, ops, [0.0], tol=1e-9, **_kwargs())
    rho = qt.rand_dm(2, seed=1)
    assert (_dissipator(ref, rho) - _dissipator(near, rho)).norm() < 1e-10


def test_collapse_operators_without_coupling_are_empty():
    ops = kondo_bath.chain_spin_operators(1)
    assert kondo_bath.kondo_collapse_operators(
        qt.sigmaz(), ops, [0.0], **_kwargs(rho_j=0.0)) == []


def test_collapse_operators_are_covariant_under_a_unitary_change_of_basis():
    # The same dimer with H and the spin operators rotated by one unitary.
    ops = kondo_bath.chain_spin_operators(2)
    ham = _dimer_hamiltonian(1.0)
    c_ops = kondo_bath.kondo_collapse_operators(
        ham, ops, [0.0, 1.0], **_kwargs())
    rotation = qt.Qobj(
        np.diag(np.exp(1j * np.arange(4))), dims=ham.dims)
    c_rot = kondo_bath.kondo_collapse_operators(
        rotation * ham * rotation.dag(),
        [tuple(rotation * o * rotation.dag() for o in site) for site in ops],
        [0.0, 1.0], **_kwargs())
    rho = qt.rand_dm([2, 2], seed=2)
    rho_rot = rotation * rho * rotation.dag()
    lhs = qt.vector_to_operator(_dissipator(c_ops, rho))
    rhs = qt.vector_to_operator(_dissipator(c_rot, rho_rot))
    assert (rotation * lhs * rotation.dag() - rhs).norm() < 1e-10


# --- level shifts ------------------------------------------------------------

def test_level_shift_of_a_spin_half_reduces_the_splitting_as_in_eq_22():
    delta0, bandwidth, kt = 2.0, 100.0, 0.5
    ops = kondo_bath.chain_spin_operators(1)
    ham = delta0 / 2 * qt.sigmaz()
    shifts = kondo_bath.level_shifts(
        ham, ops, rho_J=RHO_J, kT=kt, bandwidth=bandwidth)
    log = math.log(2 * bandwidth / (math.pi * kt))
    renormalized = delta0 + shifts[1] - shifts[0]
    assert renormalized == pytest.approx(
        delta0 * (1 - RHO_J ** 2 / 4 * log), rel=1e-12)


def test_shifted_transitions_are_antisymmetric_with_zero_diagonal():
    ops = kondo_bath.chain_spin_operators(1, 1.0)
    ham = -1.0 * ops[0][2] * ops[0][2] + 0.2 * (
        ops[0][0] * ops[0][0] - ops[0][1] * ops[0][1])
    matrix = kondo_bath.shifted_transitions(
        ham, ops, rho_J=0.2, kT=0.1, bandwidth=50.0)
    np.testing.assert_allclose(matrix, -matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)


# --- Negative cases ----------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan"), float("inf")])
def test_validate_rho_j_outside_unit_interval_raises(bad):
    with pytest.raises(ValueError, match="rho_J"):
        kondo_bath.validate_bath_parameters(
            rho_J=bad, kT=1.0, k_F=1.0, dim=3, tol=1e-9)


def test_validate_large_rho_j_warns_that_perturbation_theory_fails():
    with pytest.warns(UserWarning, match="weak coupling"):
        kondo_bath.validate_bath_parameters(
            rho_J=0.6, kT=1.0, k_F=1.0, dim=3, tol=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_validate_non_positive_temperature_raises(bad):
    with pytest.raises(ValueError, match="kT"):
        kondo_bath.validate_bath_parameters(
            rho_J=0.1, kT=bad, k_F=1.0, dim=3, tol=1e-9)


@pytest.mark.parametrize("bad", [0.0, -2.0, float("nan")])
def test_validate_non_positive_fermi_wave_number_raises(bad):
    with pytest.raises(ValueError, match="k_F"):
        kondo_bath.validate_bath_parameters(
            rho_J=0.1, kT=1.0, k_F=bad, dim=3, tol=1e-9)


@pytest.mark.parametrize("bad", [0, 4, -1, 2.5])
def test_validate_unsupported_dimension_raises(bad):
    with pytest.raises(ValueError, match="dim"):
        kondo_bath.validate_bath_parameters(
            rho_J=0.1, kT=1.0, k_F=1.0, dim=bad, tol=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1e-9, float("nan")])
def test_validate_non_positive_tolerance_raises(bad):
    with pytest.raises(ValueError, match="tol"):
        kondo_bath.validate_bath_parameters(
            rho_J=0.1, kT=1.0, k_F=1.0, dim=3, tol=bad)


def test_pair_kernel_unknown_dimension_raises():
    with pytest.raises(ValueError, match="dim"):
        kondo_bath.pair_kernel(4, np.array([1.0]))


def test_pair_kernel_negative_distance_raises():
    with pytest.raises(ValueError, match="non-negative"):
        kondo_bath.pair_kernel(3, np.array([-1.0]))


def test_bath_rate_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="kT"):
        kondo_bath.bath_rate(1.0, 0.0)


@pytest.mark.parametrize("n_sites", [0, -1, 1.5])
def test_chain_spin_operators_invalid_site_count_raises(n_sites):
    with pytest.raises(ValueError, match="n_sites"):
        kondo_bath.chain_spin_operators(n_sites)


@pytest.mark.parametrize("spin", [0.0, 0.3, 1.25, -0.5])
def test_chain_spin_operators_invalid_spin_raises(spin):
    with pytest.raises(ValueError, match="spin"):
        kondo_bath.chain_spin_operators(1, spin)


def test_kernel_matrix_empty_or_non_finite_positions_raise():
    with pytest.raises(ValueError, match="positions"):
        kondo_bath.kernel_matrix(np.array([]), k_F=1.0, dim=3)
    with pytest.raises(ValueError, match="positions"):
        kondo_bath.kernel_matrix(np.array([0.0, np.nan]), k_F=1.0, dim=3)


def test_rates_positions_count_must_match_the_number_of_sites():
    ops = kondo_bath.chain_spin_operators(2)
    with pytest.raises(ValueError, match="positions"):
        kondo_bath.kondo_rates(
            _dimer_hamiltonian(1.0), ops, [0.0], **_kwargs())


def test_rates_spin_operators_must_fit_the_hamiltonian():
    ops = kondo_bath.chain_spin_operators(1)
    with pytest.raises(ValueError, match="shape"):
        kondo_bath.kondo_rates(
            _dimer_hamiltonian(1.0), ops * 2, [0.0, 1.0], **_kwargs())


def test_rates_site_must_provide_three_spin_components():
    ops = kondo_bath.chain_spin_operators(1)
    with pytest.raises(ValueError, match="three"):
        kondo_bath.kondo_rates(
            qt.sigmaz(), [ops[0][:2]], [0.0], **_kwargs())


def test_rates_non_hermitian_hamiltonian_raises():
    ops = kondo_bath.chain_spin_operators(1)
    with pytest.raises(ValueError, match="Hermitian"):
        kondo_bath.kondo_rates(qt.sigmap(), ops, [0.0], **_kwargs())


def test_rates_non_hermitian_spin_operator_raises():
    sx, sy, sz = kondo_bath.chain_spin_operators(1)[0]
    with pytest.raises(ValueError, match="Hermitian"):
        kondo_bath.kondo_rates(
            qt.sigmaz(), [(sx, sy, 2j * sz)], [0.0], **_kwargs())


def test_rates_states_that_are_not_eigenvectors_raise():
    ops = kondo_bath.chain_spin_operators(1)
    states = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
    with pytest.raises(ValueError, match="eigenvectors"):
        kondo_bath.kondo_rates(
            qt.sigmaz(), ops, [0.0], states=states, **_kwargs())


def test_rates_states_with_wrong_shape_raise():
    ops = kondo_bath.chain_spin_operators(1)
    with pytest.raises(ValueError, match="states"):
        kondo_bath.kondo_rates(
            qt.sigmaz(), ops, [0.0], states=np.eye(3), **_kwargs())


def test_collapse_operators_propagate_parameter_errors():
    ops = kondo_bath.chain_spin_operators(1)
    with pytest.raises(ValueError, match="kT"):
        kondo_bath.kondo_collapse_operators(
            qt.sigmaz(), ops, [0.0], **_kwargs(kt=-1.0))


def test_level_shifts_bandwidth_not_much_larger_than_kt_raises():
    ops = kondo_bath.chain_spin_operators(1)
    with pytest.raises(ValueError, match="bandwidth"):
        kondo_bath.level_shifts(
            qt.sigmaz(), ops, rho_J=0.1, kT=1.0, bandwidth=1.0)


def test_level_shifts_for_several_sites_raise():
    ops = kondo_bath.chain_spin_operators(2)
    with pytest.raises(ValueError, match="single"):
        kondo_bath.level_shifts(
            _dimer_hamiltonian(1.0), ops, rho_J=0.1, kT=0.1, bandwidth=10.0)
