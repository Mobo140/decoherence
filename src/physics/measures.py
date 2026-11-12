"""Scalar measures of a density matrix.

All functions take the density matrix as an array (``measures`` that also
accept a ``qutip.Qobj`` say so). Floating-point operations are kept in the
order they had when the measures lived in the simulator, because trajectory
caches and golden outputs are fingerprinted exactly.
"""
from __future__ import annotations

import numpy as np

from src.physics import paulis

# sigma_y x sigma_y: the spin-flip operator of Wootters' concurrence.
_SIGMA_YY = np.kron(paulis.SIGMA_Y, paulis.SIGMA_Y)


def coherence_l1(rho: np.ndarray) -> float:
    """Computes the l1 coherence in the basis ``rho`` is written in.

    Args:
        rho: Density matrix, shape (d, d).

    Returns:
        Sum of the absolute values of the off-diagonal elements.
    """
    return np.sum(np.abs(rho - np.diag(np.diag(rho))))


def purity(rho: np.ndarray) -> float:
    """Computes the purity Tr(rho^2).

    Args:
        rho: Density matrix, shape (d, d).

    Returns:
        Purity, 1 for a pure state and 1/d for the maximally mixed one.
    """
    return np.real(np.trace(rho @ rho))


def energy_basis_coherence_l1(
    rho: np.ndarray, eigenvectors: np.ndarray
) -> float:
    """Computes the l1 coherence in the eigenbasis of a Hamiltonian.

    Unitary evolution only rotates the phases of rho_ij in that basis, so
    the decay of this quantity is set by dissipation alone.

    Args:
        rho: Density matrix, shape (d, d), in the computational basis.
        eigenvectors: Unitary whose columns are the eigenvectors of the
            Hamiltonian (``numpy.linalg.eigh(H)[1]``).

    Returns:
        Sum of the absolute values of the off-diagonal elements of
        ``U^dagger rho U``.
    """
    rotated = eigenvectors.conj().T @ rho @ eigenvectors
    return float(np.abs(rotated).sum() - np.abs(np.diag(rotated)).sum())


def wootters_lambda(rho) -> float:
    """Computes l1 - l2 - l3 - l4 of Wootters' concurrence formula.

    The l_i are the square roots of the eigenvalues of
    ``rho (sy x sy) rho* (sy x sy)`` in decreasing order. The value is
    negative for separable states and changes sign smoothly where the
    concurrence reaches zero, which is why it is used to interpolate the
    time of entanglement sudden death.

    Args:
        rho: Two-qubit density matrix, an array or a ``qutip.Qobj``.

    Returns:
        The unclipped Wootters lambda.
    """
    matrix = rho.full() if hasattr(rho, "full") else np.asarray(rho)
    product = matrix @ _SIGMA_YY @ matrix.conj() @ _SIGMA_YY
    eigenvalues = np.sort(np.linalg.eigvals(product).real)[::-1]
    lam = np.sqrt(np.clip(eigenvalues, 0.0, None))
    return float(lam[0] - lam[1] - lam[2] - lam[3])


def concurrence(rho) -> float:
    """Computes the Wootters concurrence max(0, wootters_lambda(rho)).

    Args:
        rho: Two-qubit density matrix, an array or a ``qutip.Qobj``.

    Returns:
        Concurrence in [0, 1].
    """
    return max(0.0, wootters_lambda(rho))
