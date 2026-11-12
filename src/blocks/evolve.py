"""Evolution of the density matrix under the Lindblad master equation.

``master_equation`` is the only place in the code base that calls
``qutip.mesolve``. It returns the states and nothing derived from them, so
observables and targets can change without a new simulation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip as qt

from src.blocks.channels import Channel, to_mesolve
from src.blocks.system import System


@dataclass(frozen=True, eq=False)
class States:
    """The density matrix along a time grid.

    Attributes:
        times: Time points, shape (n_times,).
        rho: Density matrix at each time point, as dense arrays.
    """

    times: np.ndarray
    rho: list[np.ndarray]


def _check_inputs(
    system: System, rho0: qt.Qobj, times: np.ndarray
) -> np.ndarray:
    """Returns the time grid as floats; raises ValueError if invalid."""
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(
            f"times must be a non-empty 1-D array, got shape {grid.shape}.")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("times must be finite and strictly increasing.")
    if rho0.shape != system.H.shape:
        raise ValueError(
            f"The initial state has shape {rho0.shape} but the "
            f"Hamiltonian has shape {system.H.shape}.")
    return grid


def master_equation(
    system: System,
    channels: Sequence[Channel],
    rho0: qt.Qobj,
    times: np.ndarray,
) -> States:
    """Solves the Lindblad master equation.

    Args:
        system: The Hamiltonian.
        channels: Dissipation channels; may be empty.
        rho0: Initial density matrix.
        times: Time points of the output, strictly increasing.

    Returns:
        The density matrix at every time point.

    Raises:
        ValueError: If `times` is empty, not 1-D, not finite or not
            strictly increasing, if `rho0` does not fit the Hamiltonian, or
            if a channel operator does not.
    """
    grid = _check_inputs(system, rho0, times)
    for channel in channels:
        if channel.operator.shape != system.H.shape:
            raise ValueError(
                f"A channel operator has shape {channel.operator.shape} "
                f"but the Hamiltonian has shape {system.H.shape}.")
    result = qt.mesolve(
        system.H,
        rho0,
        grid,
        to_mesolve(channels),
        e_ops=[],
        options={"nsteps": 10000},
    )
    return States(times=grid, rho=[state.full() for state in result.states])


BASES = ("z",)


@dataclass(frozen=True, eq=False)
class Record:
    """The outcomes of a sequence of projective measurements.

    Attributes:
        times: Measurement times, shape (n_times,).
        outcomes: Outcome of every qubit at every time, integers of shape
            (n_times, n_qubits); in the z basis 0 is sigma_z = +1 (the
            first basis state of QuTiP) and 1 is sigma_z = -1. Qubit 0 is
            the first tensor factor.
        rho_after: The post-measurement density matrix at every time, or
            None if it was not kept.
    """

    times: np.ndarray
    outcomes: np.ndarray
    rho_after: list[np.ndarray] | None = None


def _bits(index: int, n_qubits: int) -> list[int]:
    """Returns the outcome of each qubit for a basis-state index."""
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


def sequential_record(
    system: System,
    channels: Sequence[Channel],
    rho0: qt.Qobj,
    times: np.ndarray,
    basis: Literal["z"],
    rng: np.random.Generator,
    keep_rho: bool = False,
) -> Record:
    """Evolves and measures projectively at every time (direction C).

    The state is measured at ``times[0]``. Between consecutive times it
    evolves under the master equation (`master_equation` on the interval,
    with absolute times, so time-dependent rates are right), then it is
    measured. Each measurement draws an outcome from the Born probabilities
    of the computational basis and collapses the state to it. Only the z
    basis is implemented.

    Args:
        system: The Hamiltonian.
        channels: Dissipation channels; may be empty.
        rho0: Initial density matrix.
        times: Measurement times, strictly increasing.
        basis: ``"z"``, the computational basis of every qubit.
        rng: Source of the outcomes.
        keep_rho: Keep the post-measurement states in the record.

    Returns:
        The record.

    Raises:
        ValueError: If `basis` is not supported, `times` is invalid, or
            `rho0` or a channel does not fit the Hamiltonian.
        TypeError: If `rng` is not a Generator.
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {rng!r}.")
    grid = _check_inputs(system, rho0, times)
    dim = system.H.shape[0]
    outcomes = np.zeros((len(grid), system.n_qubits), dtype=np.int64)
    kept = []
    rho = rho0
    for k, t in enumerate(grid):
        if k > 0:
            rho = qt.Qobj(
                master_equation(
                    system, channels, rho, grid[k - 1:k + 1]).rho[-1],
                dims=rho0.dims)
        p = np.clip(np.real(np.diag(rho.full())), 0.0, None)
        if not p.sum() > 0:
            raise ValueError("The density matrix gives no outcome.")
        index = int(rng.choice(dim, p=p / p.sum()))
        outcomes[k] = _bits(index, system.n_qubits)
        projector = qt.projection(dim, index, index)
        projector.dims = rho0.dims
        rho = (projector * rho * projector) / p[index]
        if keep_rho:
            kept.append(rho.full())
    return Record(
        times=grid, outcomes=outcomes, rho_after=kept if keep_rho else None)
