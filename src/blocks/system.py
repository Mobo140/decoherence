"""The physical system: Hamiltonian, spin operators and site positions.

Every later stage (channels, initial state, evolution) takes a ``System``
instead of a class hierarchy, so a new system is one more builder function.
The builders reuse the Hamiltonians of ``lindblad_systems``, so the
published numbers do not depend on which code path built them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip as qt

from src.physics import kondo_bath, lindblad_systems


@dataclass(frozen=True, eq=False)
class System:
    """A closed quantum system of spins.

    Attributes:
        H: Hamiltonian on the full space.
        spin_ops: For each site the tuple (S^x, S^y, S^z) acting on the
            full space.
        positions: Site positions, shape (n_sites,) or (n_sites, m), in the
            lattice unit.
        n_qubits: Number of sites.
    """

    H: qt.Qobj
    spin_ops: tuple
    positions: np.ndarray
    n_qubits: int

    def __post_init__(self) -> None:
        """Validates the fields.

        Raises:
            ValueError: If `n_qubits` is not a positive integer, `H` is not
                a square operator of dimension 2^n_qubits, `spin_ops` is
                not one (S^x, S^y, S^z) triple of that shape per site, or
                `positions` is not a finite array with one row per site.
        """
        n = self.n_qubits
        if int(n) != n or n < 1:
            raise ValueError(
                f"n_qubits must be a positive integer, got {n!r}.")
        dim = 2 ** int(n)
        if self.H.shape != (dim, dim):
            raise ValueError(
                f"H must have shape {(dim, dim)} for {n} qubits, "
                f"got {self.H.shape}.")
        if len(self.spin_ops) != n or any(
                len(site) != 3 or any(op.shape != self.H.shape
                                      for op in site)
                for site in self.spin_ops):
            raise ValueError(
                f"spin_ops must hold one (Sx, Sy, Sz) triple of shape "
                f"{self.H.shape} for each of the {n} sites.")
        positions = np.asarray(self.positions, dtype=float)
        if (positions.ndim not in (1, 2) or positions.shape[0] != n
                or not np.all(np.isfinite(positions))):
            raise ValueError(
                f"positions must be a finite array with {n} rows, "
                f"got shape {positions.shape}.")


def _lattice(n_qubits: int) -> np.ndarray:
    """Returns the positions of a chain, one lattice unit apart."""
    return np.arange(n_qubits, dtype=float)


def single_qubit(omega: float) -> System:
    """Builds a single qubit with H = omega * sigma_z.

    Args:
        omega: Frequency.

    Returns:
        The system with spin-1/2 operators S = sigma / 2.

    Raises:
        ValueError: If `omega` is not finite.
    """
    if not math.isfinite(omega):
        raise ValueError(f"omega must be finite, got {omega!r}.")
    return System(
        H=omega * qt.sigmaz(),
        spin_ops=tuple(kondo_bath.chain_spin_operators(1)),
        positions=_lattice(1),
        n_qubits=1,
    )


def two_qubit(
    J: float, interaction: Literal["XXZ", "TFIM"]
) -> System:
    """Builds two coupled qubits.

    Args:
        J: Coupling strength.
        interaction: 'XXZ' for J (XX + YY + ZZ), 'TFIM' for
            J XX + (Z1 + Z2).

    Returns:
        The system, the qubits one lattice unit apart.

    Raises:
        ValueError: If `J` is not finite or `interaction` is unknown.
    """
    if not math.isfinite(J):
        raise ValueError(f"J must be finite, got {J!r}.")
    return System(
        H=lindblad_systems.two_qubit_hamiltonian(J, interaction),
        spin_ops=tuple(kondo_bath.chain_spin_operators(2)),
        positions=_lattice(2),
        n_qubits=2,
    )
