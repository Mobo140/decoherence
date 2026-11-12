"""blocks.system: the System dataclass and the single/two-qubit builders."""
import math

import numpy as np
import pytest
import qutip as qt

from src.blocks import system
from src.physics import kondo_bath, lindblad_systems

NAN = float("nan")
INF = float("inf")


def _ops(n):
    return tuple(kondo_bath.chain_spin_operators(n))


# --- Negative cases ---


@pytest.mark.parametrize("omega", [NAN, INF, -INF])
def test_single_qubit_non_finite_omega_raises_value_error(omega):
    with pytest.raises(ValueError, match="omega must be finite"):
        system.single_qubit(omega)


@pytest.mark.parametrize("value", [NAN, INF])
def test_two_qubit_non_finite_coupling_raises_value_error(value):
    with pytest.raises(ValueError, match="J must be finite"):
        system.two_qubit(value, "XXZ")


@pytest.mark.parametrize("interaction", ["xxz", "", "Heisenberg", None])
def test_two_qubit_unknown_interaction_raises_value_error(interaction):
    with pytest.raises(ValueError, match="Unknown interaction type"):
        system.two_qubit(0.5, interaction)


@pytest.mark.parametrize("n_qubits", [0, -1, 1.5])
def test_system_invalid_qubit_count_raises_value_error(n_qubits):
    with pytest.raises(ValueError, match="n_qubits must be a positive"):
        system.System(qt.sigmaz(), _ops(1), np.zeros(1), n_qubits)


def test_system_hamiltonian_of_the_wrong_shape_raises_value_error():
    with pytest.raises(ValueError, match="H must have shape"):
        system.System(qt.sigmaz(), _ops(2), np.arange(2.0), 2)


def test_system_wrong_number_of_spin_sites_raises_value_error():
    with pytest.raises(ValueError, match="spin_ops must hold"):
        system.System(qt.sigmaz(), (), np.zeros(1), 1)


def test_system_spin_triple_of_the_wrong_shape_raises_value_error():
    with pytest.raises(ValueError, match="spin_ops must hold"):
        system.System(qt.sigmaz(), (_ops(2)[0],), np.zeros(1), 1)


def test_system_incomplete_spin_triple_raises_value_error():
    with pytest.raises(ValueError, match="spin_ops must hold"):
        system.System(qt.sigmaz(), (_ops(1)[0][:2],), np.zeros(1), 1)


@pytest.mark.parametrize("positions", [
    np.zeros(2), np.zeros((1, 2, 2)), np.array([NAN])])
def test_system_invalid_positions_raise_value_error(positions):
    with pytest.raises(ValueError, match="positions must be"):
        system.System(qt.sigmaz(), _ops(1), positions, 1)


def test_system_is_frozen():
    sys = system.single_qubit(1.0)
    with pytest.raises(AttributeError):
        sys.n_qubits = 2


# --- Positive cases ---


def test_single_qubit_hamiltonian_is_omega_sigma_z():
    sys = system.single_qubit(0.7)
    legacy = lindblad_systems.SingleQubitSystem(omega=0.7, gamma_const=0.0)
    np.testing.assert_array_equal(sys.H.full(), legacy.H.full())
    assert sys.n_qubits == 1 and len(sys.spin_ops) == 1


@pytest.mark.parametrize("interaction", ["XXZ", "TFIM"])
@pytest.mark.parametrize("J", [0.0, 0.5, -1.3])
def test_two_qubit_hamiltonian_equals_the_legacy_one(interaction, J):
    sys = system.two_qubit(J, interaction)
    legacy = lindblad_systems.TwoQubitSystem(
        J=J, interaction_type=interaction, gamma_const=0.0)
    np.testing.assert_array_equal(sys.H.full(), legacy.H.full())
    assert sys.n_qubits == 2 and len(sys.spin_ops) == 2


def test_positions_are_one_lattice_unit_apart():
    np.testing.assert_array_equal(
        system.two_qubit(0.5, "XXZ").positions, [0.0, 1.0])
    np.testing.assert_array_equal(
        system.single_qubit(1.0).positions, [0.0])


def test_spin_operators_are_spin_half_on_the_full_space():
    sys = system.two_qubit(0.5, "XXZ")
    sx, sy, sz = sys.spin_ops[1]
    np.testing.assert_allclose(
        sz.full(), np.kron(np.eye(2), np.diag([0.5, -0.5])))
    assert sx.shape == sy.shape == (4, 4)
    assert math.isclose(((sx * sy - sy * sx) - 1j * sz).norm(), 0.0,
                        abs_tol=1e-12)
