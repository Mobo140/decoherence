"""Integration of the Lindblad master equation with QuTiP.

The integration itself is ``blocks.evolve.master_equation``; this module
adapts the Lindblad system classes to it and evaluates observables.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import qutip as qt

from src.blocks import channels, evolve
from src.blocks import system as blocks_system
from src.physics import kondo_bath, lindblad_systems
from src.physics import observables as state_observables

_System = lindblad_systems.SingleQubitSystem | lindblad_systems.TwoQubitSystem
_SINGLE_QUBIT_OBSERVABLES = ["sigma_x", "sigma_y", "sigma_z"]
_TWO_QUBIT_OBSERVABLES = [
    "sigma_x1", "sigma_y1", "sigma_z1",
    "sigma_x2", "sigma_y2", "sigma_z2",
]


def _block_system(system: _System) -> blocks_system.System:
    """Wraps the Hamiltonian of a Lindblad system as a block ``System``."""
    n_qubits = 1 if isinstance(
        system, lindblad_systems.SingleQubitSystem) else 2
    return blocks_system.System(
        H=system.H,
        spin_ops=tuple(kondo_bath.chain_spin_operators(n_qubits)),
        positions=np.arange(n_qubits, dtype=float),
        n_qubits=n_qubits,
    )


def _channels(system: _System) -> list[channels.Channel]:
    """Builds the channels of a Lindblad system.

    Args:
        system: The Lindblad system.

    Returns:
        One channel per jump operator, with the constant rate if the system
        has one and ``system.get_gamma`` otherwise.
    """
    rate = (system.gamma_const if system.gamma_const is not None
            else system.get_gamma)
    return [channels.Channel(L, rate) for L in system.L_list]


def _collapse_operators(system: _System) -> list:
    """Builds the collapse operators for ``qutip.mesolve``.

    Args:
        system: The Lindblad system.

    Returns:
        Operators scaled by sqrt(gamma) for a constant rate, otherwise
        [operator, sqrt(gamma(t))] pairs for QuTiP's time-dependent format.
    """
    return channels.to_mesolve(_channels(system))


def trajectories(
    density_matrices: list[np.ndarray],
    observable_names: list[str],
    compute_observables: Callable[[np.ndarray], dict[str, float]],
) -> dict:
    """Evaluates observables along a list of density matrices.

    Args:
        density_matrices: Density matrix at each time point.
        observable_names: Observables to record besides purity and l1
            coherence, which are always recorded.
        compute_observables: Maps a density matrix to a dictionary of
            observables.

    Returns:
        Dictionary of observable trajectories (arrays) plus
        ``density_matrices``, the input list.

    Raises:
        ValueError: If a name in `observable_names` is not produced by
            `compute_observables`.
    """
    result = {name: [] for name in observable_names}
    result["purity"] = []
    result["coherence_l1"] = []
    for rho in density_matrices:
        values = compute_observables(rho)
        unknown = [name for name in observable_names if name not in values]
        if unknown:
            raise ValueError(
                f"Unknown observables {unknown}; valid names are "
                f"{sorted(values)}.")
        for name in observable_names:
            result[name].append(values[name])
        result["purity"].append(values["purity"])
        result["coherence_l1"].append(values["coherence_l1"])

    for name in result:
        result[name] = np.array(result[name])
    result["density_matrices"] = density_matrices
    return result


def observe_single_qubit(
    density_matrices: list[np.ndarray],
    observables: list[str] | None = None,
) -> dict:
    """Evaluates the single-qubit observables along density matrices.

    Args:
        density_matrices: Density matrix at each time point.
        observables: Names of the Pauli expectations to record; defaults to
            ['sigma_x', 'sigma_y', 'sigma_z'].

    Returns:
        As `trajectories`.

    Raises:
        ValueError: As `trajectories`.
    """
    if observables is None:
        observables = list(_SINGLE_QUBIT_OBSERVABLES)
    return trajectories(
        density_matrices, observables,
        state_observables.compute_observables_single_qubit)


def observe_two_qubit(
    density_matrices: list[np.ndarray],
    observables: list[str] | None = None,
) -> dict:
    """Evaluates the two-qubit observables along density matrices.

    Args:
        density_matrices: Density matrix at each time point.
        observables: Names of the Pauli expectations to record; defaults to
            sigma_{x,y,z}1 and sigma_{x,y,z}2.

    Returns:
        As `trajectories`.

    Raises:
        ValueError: As `trajectories`.
    """
    if observables is None:
        observables = list(_TWO_QUBIT_OBSERVABLES)
    return trajectories(
        density_matrices, observables,
        state_observables.compute_observables_two_qubit)


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
    states = evolve.master_equation(
        _block_system(system), _channels(system), initial_state, times)
    return trajectories(states.rho, observable_names, compute_observables)


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
