"""Evolution of the density matrix under the Lindblad master equation.

``master_equation`` is the only place in the code base that calls
``qutip.mesolve``. It returns the states and nothing derived from them, so
observables and targets can change without a new simulation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
