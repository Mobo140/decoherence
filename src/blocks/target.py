"""Targets: the decoherence time as a function of the states.

A target maps ``States`` and the ``System`` to a ``Target`` (time and
censoring flag). Every target is the first crossing of a threshold by a
curve computed from rho(t), linearly interpolated, or the end of the grid
with ``censored=True`` if it is never crossed. Because the target is
computed from the states, changing it does not need a new simulation.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.blocks.evolve import States
from src.blocks.system import System
from src.physics import measures

# Below this the initial value is treated as zero and the threshold is
# absolute instead of relative to it.
_MIN_INITIAL_VALUE = 1e-10
# Regularises the interpolation when two neighbouring values coincide.
_INTERPOLATION_EPS = 1e-12

# The same values as CRITERIA, as a type the block catalog can read.
Criterion = Literal["coherence", "purity", "concurrence"]
CRITERIA = ("coherence", "purity", "concurrence")


@dataclass(frozen=True)
class Target:
    """The decoherence time of one trajectory.

    Attributes:
        time: The crossing time; the last time point, a lower bound, if
            the level is never reached.
        censored: True if the level is never reached.
    """

    time: float
    censored: bool


def interpolate_crossing(
    times: np.ndarray,
    values: Sequence[float],
    index: int,
    level: float,
) -> float:
    """Interpolates linearly where a curve crosses a level.

    Args:
        times: Time points.
        values: Curve values at ``times``.
        index: Index of the first value that is past the level.
        level: The level that is crossed between ``index - 1`` and ``index``.

    Returns:
        The crossing time; ``times[0]`` if the curve starts past the level.

    Raises:
        ValueError: If `index` is not a position in both `times` and
            `values`.
    """
    if not 0 <= index < min(len(times), len(values)):
        raise ValueError(
            f"index {index} is outside the {len(times)} times and "
            f"{len(values)} values.")
    if index == 0:
        return float(times[0])
    prev_val = float(values[index - 1])
    val = float(values[index])
    frac = (prev_val - level) / (prev_val - val + _INTERPOLATION_EPS)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(times[index - 1] + frac * (times[index] - times[index - 1]))


def first_crossing(
    times: np.ndarray,
    values: Sequence[float],
    level: float,
    inclusive: bool = False,
) -> tuple[float, bool]:
    """Finds the first time a decaying curve falls below a level.

    Every decoherence criterion reduces to this function.

    Args:
        times: Time points.
        values: Curve values at ``times``.
        level: The level to fall below.
        inclusive: If True, a value equal to ``level`` already counts.

    Returns:
        ``(time, censored)``. ``time`` is the interpolated crossing time.
        If the level is never reached, ``time`` is ``times[-1]``, a lower
        bound, and ``censored`` is True.

    Raises:
        ValueError: If `times` or `values` is empty, they differ in
            length, or `level` or a value is NaN.
    """
    if len(times) == 0 or len(values) == 0:
        raise ValueError("times and values must not be empty.")
    if len(times) != len(values):
        raise ValueError(
            f"times has {len(times)} points but values has {len(values)}.")
    if math.isnan(level) or np.isnan(np.asarray(values, dtype=float)).any():
        raise ValueError("level and values must not contain NaN.")
    for i, value in enumerate(values):
        crossed = value <= level if inclusive else value < level
        if crossed:
            return interpolate_crossing(times, values, i, level), False
    return float(times[-1]), True


def relative_level(values: Sequence[float], threshold: float) -> float:
    """Returns ``threshold`` times the initial value of a curve.

    Args:
        values: Curve values; the first one is the initial value.
        threshold: Fraction of the initial value, e.g. 1/e.

    Returns:
        The level, or ``threshold`` itself if the curve starts at zero.
    """
    initial = float(values[0]) if len(values) > 0 else 1.0
    return threshold * initial if initial > _MIN_INITIAL_VALUE else threshold


def sudden_death_time(
    times: np.ndarray, lam: Sequence[float]
) -> tuple[float, bool]:
    """Finds the time of entanglement sudden death.

    The first zero of the concurrence, interpolated on the unclipped
    Wootters lambda. A state that starts separable gives 0 (no usable
    windows); one that is still entangled at the end is censored.

    Args:
        times: Time points.
        lam: Unclipped Wootters lambda at ``times``.

    Returns:
        ``(time, censored)`` as in ``first_crossing``.
    """
    return first_crossing(times, lam, 0.0, inclusive=True)


def energy_basis_coherence(
    hamiltonian: np.ndarray, density_matrices: Sequence
) -> list[float]:
    """Computes the l1 coherence in the energy basis along a trajectory.

    Args:
        hamiltonian: Hamiltonian as a dense array.
        density_matrices: Density matrix at each time point, arrays or
            objects with a ``full()`` method.

    Returns:
        The coherence at each time point.

    Raises:
        ValueError: If `hamiltonian` is not a Hermitian square matrix.
    """
    hamiltonian = np.asarray(hamiltonian)
    if (hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]
            or not np.allclose(hamiltonian, hamiltonian.conj().T)):
        raise ValueError("hamiltonian must be a Hermitian square matrix.")
    eigenvectors = np.linalg.eigh(hamiltonian)[1]
    return [
        measures.energy_basis_coherence_l1(
            r.full() if hasattr(r, "full") else np.asarray(r), eigenvectors
        )
        for r in density_matrices
    ]


def _check_threshold(threshold: float) -> None:
    """Raises ValueError unless `threshold` is a number in (0, 1)."""
    if not (isinstance(threshold, (int, float)) and 0 < threshold < 1):
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}.")


def _check_states(states: States, system: System) -> None:
    """Raises ValueError if the states are empty or do not fit the system."""
    if len(states.rho) == 0 or len(states.rho) != len(states.times):
        raise ValueError(
            f"states must hold one matrix per time point, got "
            f"{len(states.rho)} matrices for {len(states.times)} times.")
    if np.shape(states.rho[0]) != system.H.shape:
        raise ValueError(
            f"The states have shape {np.shape(states.rho[0])} but the "
            f"Hamiltonian has shape {system.H.shape}.")


def _crossing(
    times: np.ndarray, values: Sequence[float], level: float
) -> Target:
    time, censored = first_crossing(times, values, level)
    return Target(time, censored)


def coherence_crossing(
    states: States,
    system: System,
    threshold: float,
    criterion: Criterion = "coherence",
) -> Target:
    """Finds the first crossing of a threshold by a decaying curve (T2).

    The levels are ``threshold`` times the initial l1 coherence or
    concurrence; for the purity, the decay from the initial purity towards
    the maximally mixed value 1 / 2^n has fallen by the fraction
    ``threshold``.

    Args:
        states: The states.
        system: The system the states live on.
        threshold: Fraction in (0, 1), e.g. 1/e.
        criterion: 'coherence', 'purity' or 'concurrence'; the last needs
            two qubits.

    Returns:
        The target.

    Raises:
        ValueError: If `criterion` is unknown or not defined for the
            system, `threshold` is not in (0, 1), the states do not fit
            the system, or a curve contains NaN.
    """
    _check_threshold(threshold)
    _check_states(states, system)
    if criterion == "coherence":
        values = [measures.coherence_l1(r) for r in states.rho]
        return _crossing(
            states.times, values, relative_level(values, threshold))
    if criterion == "purity":
        values = [measures.purity(r) for r in states.rho]
        initial = float(values[0])
        p_min = 1.0 / (2 ** system.n_qubits)
        return _crossing(
            states.times, values, initial - threshold * (initial - p_min))
    if criterion == "concurrence":
        if system.n_qubits != 2:
            raise ValueError("The concurrence criterion needs two qubits.")
        values = [measures.concurrence(r) for r in states.rho]
        return _crossing(
            states.times, values, relative_level(values, threshold))
    raise ValueError(
        f"criterion must be one of {CRITERIA}, got {criterion!r}.")


def energy_basis_crossing(
    states: States, system: System, threshold: float
) -> Target:
    """Finds the crossing of the l1 coherence in the energy basis (T2^E).

    Only the target changes: the observables fed to a model stay in the
    computational basis.

    Args:
        states: The states.
        system: The system; its Hamiltonian defines the basis.
        threshold: Fraction in (0, 1) of the initial coherence.

    Returns:
        The target.

    Raises:
        ValueError: If `threshold` is not in (0, 1), the states do not fit
            the system, or the Hamiltonian is not Hermitian.
    """
    _check_threshold(threshold)
    _check_states(states, system)
    values = energy_basis_coherence(system.H.full(), states.rho)
    return _crossing(states.times, values, relative_level(values, threshold))


def sudden_death(states: States, system: System) -> Target:
    """Finds the time of entanglement sudden death (ESD).

    Args:
        states: Two-qubit states.
        system: The two-qubit system.

    Returns:
        The target: the first zero of the concurrence, censored if the
        state is still entangled at the end.

    Raises:
        ValueError: If the system does not have two qubits or the states
            do not fit it.
    """
    if system.n_qubits != 2:
        raise ValueError("Sudden death needs two qubits.")
    _check_states(states, system)
    time, censored = sudden_death_time(
        states.times, [measures.wootters_lambda(r) for r in states.rho])
    return Target(time, censored)
