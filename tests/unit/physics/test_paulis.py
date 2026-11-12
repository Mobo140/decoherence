"""Pauli matrices are shared read-only constants."""
import numpy as np
import pytest

from src.physics import paulis


def test_pauli_algebra():
    x, y, z, eye = (
        paulis.SIGMA_X, paulis.SIGMA_Y, paulis.SIGMA_Z, paulis.IDENTITY)
    for p in (x, y, z):
        np.testing.assert_allclose(p @ p, eye)
        np.testing.assert_allclose(p, p.conj().T)
    np.testing.assert_allclose(x @ y, 1j * z)
    np.testing.assert_allclose(y @ z, 1j * x)
    np.testing.assert_allclose(z @ x, 1j * y)


def test_pauli_dict_is_keyed_by_axis():
    assert set(paulis.PAULI) == {"x", "y", "z"}
    assert paulis.PAULI["x"] is paulis.SIGMA_X
    assert paulis.PAULI["z"] is paulis.SIGMA_Z


def test_pauli_constants_are_complex_two_by_two():
    for matrix in (paulis.SIGMA_X, paulis.SIGMA_Y, paulis.SIGMA_Z,
                   paulis.IDENTITY):
        assert matrix.shape == (2, 2) and matrix.dtype == complex


# --- Negative cases ---


@pytest.mark.parametrize("name", ["SIGMA_X", "SIGMA_Y", "SIGMA_Z", "IDENTITY"])
def test_pauli_constant_cannot_be_modified_in_place(name):
    with pytest.raises(ValueError, match="read-only"):
        getattr(paulis, name)[0, 0] = 5.0


def test_pauli_dict_gives_the_same_read_only_arrays():
    with pytest.raises(ValueError, match="read-only"):
        paulis.PAULI["y"][0, 1] = 1.0
