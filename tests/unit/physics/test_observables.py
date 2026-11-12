"""Observables computed from a density matrix."""
import numpy as np
import pytest
import qutip as qt

from src.physics.measures import coherence_l1, purity
from src.physics.observables import (
    compute_observables_single_qubit,
    compute_observables_two_qubit,
)


def test_single_qubit_observables_agree_with_the_measures():
    rho = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
    obs = compute_observables_single_qubit(rho)
    assert 'sigma_x' in obs and 'sigma_y' in obs and 'sigma_z' in obs
    assert np.isclose(obs['coherence_l1'], coherence_l1(rho))
    assert np.isclose(obs['purity'], purity(rho))


def test_single_qubit_observables_of_the_bloch_axis_states():
    cases = {
        (1.0, 0.0, 0.0): qt.basis(2, 0) + qt.basis(2, 1),
        (0.0, 1.0, 0.0): qt.basis(2, 0) + 1j * qt.basis(2, 1),
        (0.0, 0.0, 1.0): qt.basis(2, 0),
        (0.0, 0.0, -1.0): qt.basis(2, 1),
    }
    for (x, y, z), ket in cases.items():
        obs = compute_observables_single_qubit(qt.ket2dm(ket.unit()).full())
        assert (obs["sigma_x"], obs["sigma_y"], obs["sigma_z"]) == (
            pytest.approx(x), pytest.approx(y), pytest.approx(z))
        assert obs["purity"] == pytest.approx(1.0)


def test_single_qubit_observable_names():
    obs = compute_observables_single_qubit(np.eye(2, dtype=complex) / 2)
    assert set(obs) == {
        "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"}
    assert obs["purity"] == pytest.approx(0.5)


def test_two_qubit_observables_of_a_product_state():
    rho = qt.ket2dm(qt.basis([2, 2], [0, 1])).full()   # |0>|1>
    obs = compute_observables_two_qubit(rho)
    assert obs["sigma_z1"] == pytest.approx(1.0)
    assert obs["sigma_z2"] == pytest.approx(-1.0)
    assert obs["sigma_x1"] == pytest.approx(0.0, abs=1e-12)
    assert obs["purity"] == pytest.approx(1.0)
    assert obs["coherence_l1"] == pytest.approx(0.0)


def test_two_qubit_observables_of_a_bell_state_have_no_local_polarisation():
    bell = (qt.basis([2, 2], [0, 0]) + qt.basis([2, 2], [1, 1])).unit()
    obs = compute_observables_two_qubit(qt.ket2dm(bell).full())
    for name in ("sigma_x1", "sigma_y1", "sigma_z1",
                 "sigma_x2", "sigma_y2", "sigma_z2"):
        assert obs[name] == pytest.approx(0.0, abs=1e-12)
    assert obs["coherence_l1"] == pytest.approx(1.0)


def test_two_qubit_observable_names_and_their_order():
    obs = compute_observables_two_qubit(np.eye(4, dtype=complex) / 4)
    assert list(obs) == [
        "sigma_x1", "sigma_y1", "sigma_z1", "sigma_x2", "sigma_y2",
        "sigma_z2", "purity", "coherence_l1"]
    assert obs["purity"] == pytest.approx(0.25)


def test_pauli_expectations_are_real_for_a_complex_density_matrix():
    rho = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    obs = compute_observables_single_qubit(rho)
    assert np.isrealobj(obs["sigma_y"]) and obs["sigma_y"] == (
        pytest.approx(-1.0))


# --- Negative cases ---


@pytest.mark.parametrize("shape", [(4, 4), (3, 3), (2, 3), (2,), (1, 2, 2)])
def test_single_qubit_observables_of_a_wrong_shape_raise_value_error(shape):
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        compute_observables_single_qubit(np.zeros(shape))


@pytest.mark.parametrize("shape", [(2, 2), (8, 8), (4, 3), (4,), (2, 4, 4)])
def test_two_qubit_observables_of_a_wrong_shape_raise_value_error(shape):
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        compute_observables_two_qubit(np.zeros(shape))


def test_two_qubit_observables_of_a_qutip_object_raise_type_error():
    # A Qobj is not an array; the caller must pass `.full()`.
    with pytest.raises(TypeError, match="incompatible dimensions"):
        compute_observables_two_qubit(qt.qeye([2, 2]) / 4)
