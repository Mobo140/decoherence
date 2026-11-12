"""Decoherence measures: purity, l1-coherence, concurrence."""
import numpy as np
import pytest
import qutip as qt

from src.physics.measures import (
    coherence_l1,
    concurrence,
    energy_basis_coherence_l1,
    purity,
    wootters_lambda,
)

NAN = float("nan")


def _bell() -> qt.Qobj:
    return (qt.basis(4, 1) + qt.basis(4, 2)).unit()


def test_purity_and_coherence_of_pure_and_mixed_states():
    # Pure state |0><0|
    rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    assert np.isclose(purity(rho_pure), 1.0)
    assert np.isclose(coherence_l1(rho_pure), 0.0)

    # Maximally mixed state
    rho_mix = 0.5 * np.eye(2, dtype=complex)
    assert np.isclose(purity(rho_mix), 0.5)
    assert np.isclose(coherence_l1(rho_mix), 0.0)


def test_coherence_l1_of_plus_state_is_one():
    plus = np.full((2, 2), 0.5, dtype=complex)
    assert coherence_l1(plus) == pytest.approx(1.0)
    assert purity(plus) == pytest.approx(1.0)


def test_coherence_l1_sums_the_absolute_off_diagonal_entries():
    rho = np.array([[0.5, 0.1j, 0.0], [-0.1j, 0.3, 0.2], [0.0, 0.2, 0.2]])
    assert coherence_l1(rho) == pytest.approx(2 * (0.1 + 0.2))


def test_purity_of_a_two_qubit_maximally_mixed_state_is_a_quarter():
    assert purity(np.eye(4, dtype=complex) / 4) == pytest.approx(0.25)


def test_purity_is_real_for_a_complex_density_matrix():
    rho = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    assert np.isrealobj(purity(rho)) and purity(rho) == pytest.approx(1.0)


def test_energy_basis_coherence_in_the_computational_basis_equals_l1():
    rho = np.array([[0.6, 0.3 - 0.1j], [0.3 + 0.1j, 0.4]])
    assert energy_basis_coherence_l1(rho, np.eye(2)) == pytest.approx(
        coherence_l1(rho))


def test_energy_basis_coherence_is_zero_for_a_stationary_state():
    # |+><+| is diagonal in the eigenbasis of sigma_x.
    plus = np.full((2, 2), 0.5, dtype=complex)
    eigenvectors = np.linalg.eigh(qt.sigmax().full())[1]
    assert energy_basis_coherence_l1(plus, eigenvectors) == pytest.approx(
        0.0, abs=1e-12)
    assert isinstance(energy_basis_coherence_l1(plus, eigenvectors), float)


def test_energy_basis_coherence_ignores_the_unitary_phase_rotation():
    h = qt.sigmaz().full()
    eigenvectors = np.linalg.eigh(h)[1]
    plus = np.full((2, 2), 0.5, dtype=complex)
    rotated = qt.Qobj(plus)
    for t in (0.0, 0.7, 2.3):
        u = (-1j * h * t).tolist()
        evolved = qt.Qobj(plus).transform(qt.Qobj(u).expm())
        assert energy_basis_coherence_l1(
            evolved.full(), eigenvectors) == pytest.approx(1.0)
    assert rotated.shape == (2, 2)


def test_concurrence_of_known_states():
    bell = _bell()
    assert abs(concurrence(bell * bell.dag()) - 1.0) < 1e-12
    prod = qt.tensor(qt.basis(2, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    assert concurrence(prod * prod.dag()) < 1e-12
    a, b = np.cos(0.3), np.sin(0.3)
    psi = a * qt.basis(4, 0) + b * qt.basis(4, 3)
    assert abs(concurrence(psi * psi.dag()) - 2 * a * b) < 1e-12
    for p in (0.2, 0.5, 0.9):                       # Werner states
        w = p * (bell * bell.dag()).full() + (1 - p) * np.eye(4) / 4
        assert abs(concurrence(w) - max(0.0, (3 * p - 1) / 2)) < 1e-12
        assert abs(wootters_lambda(w) - (3 * p - 1) / 2) < 1e-12   # unclipped


def test_concurrence_accepts_arrays_and_qutip_objects_alike():
    bell = _bell()
    rho = bell * bell.dag()
    assert concurrence(rho) == pytest.approx(concurrence(rho.full()))
    assert wootters_lambda(rho) == pytest.approx(wootters_lambda(rho.full()))


def test_concurrence_is_clipped_at_zero_but_lambda_is_not():
    maximally_mixed = np.eye(4, dtype=complex) / 4
    assert concurrence(maximally_mixed) == 0.0
    assert wootters_lambda(maximally_mixed) == pytest.approx(-0.5)


def test_concurrence_of_the_zero_matrix_is_zero():
    assert concurrence(np.zeros((4, 4))) == 0.0


# --- Negative cases ---


@pytest.mark.parametrize("measure", [coherence_l1, purity])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3,), (2, 2, 2), (0, 0)])
def test_measure_of_a_non_square_or_empty_array_raises_value_error(
        measure, shape):
    with pytest.raises(ValueError, match="non-empty square matrix"):
        measure(np.zeros(shape))


@pytest.mark.parametrize("measure", [coherence_l1, purity])
def test_measure_of_a_scalar_raises_value_error(measure):
    with pytest.raises(ValueError, match="non-empty square matrix"):
        measure(np.float64(1.0))


def test_coherence_of_a_nan_matrix_is_nan():
    # Documented by the formula: NaN propagates, nothing hides it.
    assert np.isnan(coherence_l1(np.full((2, 2), NAN)))
    assert np.isnan(purity(np.full((2, 2), NAN)))


def test_purity_does_not_check_the_trace():
    # Debatable, kept: an unnormalised matrix gives Tr(rho^2) as is.
    assert purity(3 * np.eye(2)) == pytest.approx(18.0)


def test_energy_basis_coherence_with_mismatched_eigenvectors_raises():
    with pytest.raises(ValueError, match="eigenvectors have shape"):
        energy_basis_coherence_l1(np.eye(2, dtype=complex), np.eye(3))


def test_energy_basis_coherence_of_a_non_square_rho_raises():
    with pytest.raises(ValueError, match="non-empty square matrix"):
        energy_basis_coherence_l1(np.zeros((2, 3)), np.eye(2))


@pytest.mark.parametrize("measure", [wootters_lambda, concurrence])
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (8, 8), (4, 3), (4,)])
def test_wootters_measure_of_a_matrix_that_is_not_4x4_raises(measure, shape):
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        measure(np.zeros(shape))


def test_wootters_lambda_of_a_one_qubit_qutip_state_raises():
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        wootters_lambda(qt.ket2dm(qt.basis(2, 0)))


def test_wootters_lambda_of_a_nan_matrix_raises_linalg_error():
    with pytest.raises(np.linalg.LinAlgError):
        wootters_lambda(np.full((4, 4), NAN))
