"""Integration of the Lindblad master equation with QuTiP."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import qutip as qt

from src.physics import lindblad_systems
from src.physics import observables as state_observables

_System = lindblad_systems.SingleQubitSystem | lindblad_systems.TwoQubitSystem
_SINGLE_QUBIT_OBSERVABLES = ["sigma_x", "sigma_y", "sigma_z"]
_TWO_QUBIT_OBSERVABLES = [
    "sigma_x1", "sigma_y1", "sigma_z1",
    "sigma_x2", "sigma_y2", "sigma_z2",
]


def _collapse_operators(system: _System) -> list:
    """Builds the collapse operators for ``qutip.mesolve``.

    Args:
        system: The Lindblad system.

    Returns:
        Operators scaled by sqrt(gamma) for a constant rate, otherwise
        [operator, sqrt(gamma(t))] pairs for QuTiP's time-dependent format.
    """
    if system.gamma_const is not None:
        return [
            np.sqrt(rate) * L
            for L, rate in system.get_lindblad_operators(0.0)
        ]

    def sqrt_gamma(t, args):
        """Returns sqrt(gamma(t)) for QuTiP's callable coefficient format.

        Args:
            t: Time.
            args: Unused; required by the QuTiP coefficient signature.
        """
        gamma_t = system.get_gamma(t)
        return np.sqrt(gamma_t) if gamma_t > 0 else 0.0

    return [[L, sqrt_gamma] for L in system.L_list]


def _check_inputs(
    system: _System,
    initial_state: qt.Qobj,
    times: np.ndarray,
) -> None:
    """Raises ValueError if the time grid or the initial state is invalid."""
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(
            f"times must be a non-empty 1-D array, got shape {grid.shape}.")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("times must be finite and strictly increasing.")
    if initial_state.shape != system.H.shape:
        raise ValueError(
            f"The initial state has shape {initial_state.shape} but the "
            f"Hamiltonian has shape {system.H.shape}.")


def _simulate(
    system: _System,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observable_names: list[str],
    compute_observables: Callable[[np.ndarray], dict[str, float]],
) -> dict:
    """Solves the master equation and evaluates observables along it.

    Args:
        system: The Lindblad system.
        initial_state: Initial density matrix.
        times: Time points of the output.
        observable_names: Observables to record besides purity and l1
            coherence, which are always recorded.
        compute_observables: Maps a density matrix to a dictionary of
            observables.

    Returns:
        Dictionary of observable trajectories (arrays) plus
        ``density_matrices``, the list of density matrices as arrays.

    Raises:
        ValueError: If `times` is empty, not 1-D, not finite or not strictly
            increasing; if `initial_state` does not fit the system; or if a
            name in `observable_names` is not produced by
            `compute_observables`.
    """
    _check_inputs(system, initial_state, times)
    result = qt.mesolve(
        system.H,
        initial_state,
        times,
        _collapse_operators(system),
        e_ops=[],
        options={"nsteps": 10000},
    )
    density_matrices = [state.full() for state in result.states]

    trajectories = {name: [] for name in observable_names}
    trajectories["purity"] = []
    trajectories["coherence_l1"] = []
    for rho in density_matrices:
        values = compute_observables(rho)
        unknown = [name for name in observable_names if name not in values]
        if unknown:
            raise ValueError(
                f"Unknown observables {unknown}; valid names are "
                f"{sorted(values)}.")
        for name in observable_names:
            trajectories[name].append(values[name])
        trajectories["purity"].append(values["purity"])
        trajectories["coherence_l1"].append(values["coherence_l1"])

    for name in trajectories:
        trajectories[name] = np.array(trajectories[name])
    trajectories["density_matrices"] = density_matrices
    return trajectories


def simulate_single_qubit(
    system: lindblad_systems.SingleQubitSystem,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: list[str] | None = None,
) -> dict:
    """Simulates a single qubit with constant or time-dependent dissipation.

    Args:
        system: Single-qubit system.
        initial_state: Initial density matrix.
        times: Time points of the output.
        observables: Names of the Pauli expectations to record; defaults to
            ['sigma_x', 'sigma_y', 'sigma_z'].

    Returns:
        Dictionary of observable trajectories, including ``purity`` and
        ``coherence_l1``, and ``density_matrices``.

    Raises:
        ValueError: As `_simulate`.
    """
    if observables is None:
        observables = list(_SINGLE_QUBIT_OBSERVABLES)
    return _simulate(
        system, initial_state, times, observables,
        state_observables.compute_observables_single_qubit,
    )


def simulate_two_qubit(
    system: lindblad_systems.TwoQubitSystem,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: list[str] | None = None,
) -> dict:
    """Simulates two qubits with constant or time-dependent dissipation.

    Args:
        system: Two-qubit system.
        initial_state: Initial density matrix.
        times: Time points of the output.
        observables: Names of the Pauli expectations to record; defaults to
            sigma_{x,y,z}1 and sigma_{x,y,z}2.

    Returns:
        Dictionary of observable trajectories, including ``purity`` and
        ``coherence_l1``, and ``density_matrices``.

    Raises:
        ValueError: As `_simulate`.
    """
    if observables is None:
        observables = list(_TWO_QUBIT_OBSERVABLES)
    return _simulate(
        system, initial_state, times, observables,
        state_observables.compute_observables_two_qubit,
    )
