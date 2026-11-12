"""Pauli expectation values of a density matrix."""
from __future__ import annotations

import numpy as np

from src.physics import measures, paulis

_SIGMA_X1 = np.kron(paulis.SIGMA_X, paulis.IDENTITY)
_SIGMA_Y1 = np.kron(paulis.SIGMA_Y, paulis.IDENTITY)
_SIGMA_Z1 = np.kron(paulis.SIGMA_Z, paulis.IDENTITY)
_SIGMA_X2 = np.kron(paulis.IDENTITY, paulis.SIGMA_X)
_SIGMA_Y2 = np.kron(paulis.IDENTITY, paulis.SIGMA_Y)
_SIGMA_Z2 = np.kron(paulis.IDENTITY, paulis.SIGMA_Z)


def _expectation(rho: np.ndarray, operator: np.ndarray) -> float:
    """Returns Re Tr(rho operator)."""
    return np.real(np.trace(rho @ operator))


def compute_observables_single_qubit(rho: np.ndarray) -> dict[str, float]:
    """Computes the single-qubit observables of a density matrix.

    Args:
        rho: Density matrix, shape (2, 2).

    Returns:
        Dictionary with the Pauli expectations ``sigma_x``, ``sigma_y``,
        ``sigma_z``, the ``purity`` and the ``coherence_l1``.
    """
    return {
        "sigma_x": _expectation(rho, paulis.SIGMA_X),
        "sigma_y": _expectation(rho, paulis.SIGMA_Y),
        "sigma_z": _expectation(rho, paulis.SIGMA_Z),
        "purity": measures.purity(rho),
        "coherence_l1": measures.coherence_l1(rho),
    }


def compute_observables_two_qubit(rho: np.ndarray) -> dict[str, float]:
    """Computes the two-qubit observables of a density matrix.

    Args:
        rho: Density matrix, shape (4, 4).

    Returns:
        Dictionary with the Pauli expectations ``sigma_{x,y,z}1`` and
        ``sigma_{x,y,z}2`` of each qubit, the ``purity`` and the
        ``coherence_l1``.
    """
    return {
        "sigma_x1": _expectation(rho, _SIGMA_X1),
        "sigma_y1": _expectation(rho, _SIGMA_Y1),
        "sigma_z1": _expectation(rho, _SIGMA_Z1),
        "sigma_x2": _expectation(rho, _SIGMA_X2),
        "sigma_y2": _expectation(rho, _SIGMA_Y2),
        "sigma_z2": _expectation(rho, _SIGMA_Z2),
        "purity": measures.purity(rho),
        "coherence_l1": measures.coherence_l1(rho),
    }
