"""Initial states of a trajectory.

``random_pure`` without a generator draws from NumPy's global state, as the
published runs did, so their golden outputs stay exact; with a generator
the draw is private to the caller.
"""
from __future__ import annotations

import numpy as np
import qutip as qt

from src.blocks.system import System
from src.physics import lindblad_systems

# Parity of the two-qubit Hamiltonian the code builds,
# H = J sx1 sx2 + h (sz1 + sz2).
PARITY = qt.tensor(qt.sigmaz(), qt.sigmaz())
# Smallest norm of a projected initial state that is kept.
MIN_SECTOR_NORM = 1e-3
# Redraws of an initial state with a vanishing weight in the sector.
MAX_SECTOR_DRAWS = 100


def random_pure(
    system: System, rng: np.random.Generator | None = None
) -> qt.Qobj:
    """Draws a Haar-random pure state as a density matrix.

    Args:
        system: The system the state lives on.
        rng: Generator to draw from. If None, the legacy path with NumPy's
            global generator is used, so ``np.random.seed`` fixes the draw.

    Returns:
        The density matrix |psi><psi|.
    """
    if rng is None:
        return lindblad_systems.sample_random_pure_state(
            n_qubits=system.n_qubits)
    return lindblad_systems.sample_random_pure_state(
        n_qubits=system.n_qubits, rng=rng)


def parity_sector(
    system: System,
    sector: int,
    rng: np.random.Generator | None = None,
    *,
    min_norm: float = MIN_SECTOR_NORM,
    max_draws: int = MAX_SECTOR_DRAWS,
) -> qt.Qobj:
    """Projects a random state onto one parity sector of two qubits (R6).

    Known defect, kept because R6's recorded numbers come from it: the
    drawn state is a density matrix rho and the projector P acts on it from
    the left only, so the result is P rho / |P rho| (trace norm) instead of
    P rho P / tr(P rho P). It is not Hermitian and its trace is below one.

    A draw whose projection has a norm of at most ``min_norm`` is redrawn
    rather than renormalised: amplifying a near-zero projection would make
    the state an artefact of the discarded sector.

    Args:
        system: A two-qubit system.
        sector: Parity eigenvalue, +1 or -1.
        rng: Generator to draw from, or None for the global one.
        min_norm: Smallest projected norm that is kept.
        max_draws: Number of draws before giving up.

    Returns:
        The projected operator with unit trace norm.

    Raises:
        ValueError: If `sector` is not +1 or -1 or the system does not have
            two qubits.
        RuntimeError: If no draw falls into the sector.
    """
    if sector not in (+1, -1):
        raise ValueError("sector must be +1 or -1")
    if system.n_qubits != 2:
        raise ValueError(
            "parity_sector needs a two-qubit system, "
            f"got {system.n_qubits} qubits.")
    projector = (qt.qeye([2, 2]) + sector * PARITY) / 2
    for _ in range(max_draws):
        projected = projector * random_pure(system, rng)
        if projected.norm() > min_norm:
            return projected.unit()
    raise RuntimeError("could not draw a state inside the requested sector")
