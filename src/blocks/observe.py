"""Observation: named series computed from the states of a trajectory.

An observation function maps ``States`` to ``Observations``, a set of named
time series. Exact observations come straight from the density matrices;
the noise functions act on a trajectory's series, so a model sees what a
measurement of that one trajectory would give. ``combine`` and ``select``
assemble the series a model receives, addressed by name, never by column
position. The formulas live in ``src.physics``; this module only collects
them into series.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.blocks.evolve import States
from src.data.observation import ar1_filter
from src.physics import measures, observables, paulis

# Two-body Pauli correlators <sigma_a x sigma_b>, keyed by series name.
CORR = {
    f"corr_{a}{b}": np.kron(paulis.PAULI[a], paulis.PAULI[b])
    for a in "xyz" for b in "xyz"
}


@dataclass(frozen=True, eq=False)
class Observations:
    """Named time series of one trajectory.

    Attributes:
        times: Time points, shape (n_times,).
        series: Series by name, each of shape (n_times,); the insertion
            order is the order of the names.
    """

    times: np.ndarray
    series: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates the shapes.

        Raises:
            ValueError: If `times` is not 1-D or a series does not have one
                value per time point.
        """
        if np.ndim(self.times) != 1:
            raise ValueError(
                f"times must be 1-D, got shape {np.shape(self.times)}.")
        for name, values in self.series.items():
            if np.shape(values) != np.shape(self.times):
                raise ValueError(
                    f"Series {name!r} has shape {np.shape(values)} for "
                    f"{len(self.times)} time points.")


def _check_states(states: States, size: int | None = None) -> None:
    """Raises ValueError if the states are empty or not size x size."""
    if len(states.rho) == 0:
        raise ValueError("states must hold at least one density matrix.")
    if len(states.rho) != len(states.times):
        raise ValueError(
            f"states has {len(states.rho)} matrices for "
            f"{len(states.times)} times.")
    if size is not None and any(
            np.shape(r) != (size, size) for r in states.rho):
        raise ValueError(
            f"Expected density matrices of shape ({size}, {size}), got "
            f"{np.shape(states.rho[0])}.")


def pauli_means(states: States) -> Observations:
    """Observes the Pauli expectation values of each qubit.

    Args:
        states: States of one or two qubits.

    Returns:
        ``sigma_x, sigma_y, sigma_z`` for one qubit, or
        ``sigma_{x,y,z}1`` and ``sigma_{x,y,z}2`` for two.

    Raises:
        ValueError: If `states` is empty, or its matrices are neither 2x2
            nor 4x4.
    """
    _check_states(states)
    size = np.shape(states.rho[0])[0]
    if size == 2:
        compute = observables.compute_observables_single_qubit
        names = ["sigma_x", "sigma_y", "sigma_z"]
    elif size == 4:
        compute = observables.compute_observables_two_qubit
        names = [f"sigma_{a}{q}" for q in "12" for a in "xyz"]
    else:
        raise ValueError(
            f"pauli_means needs 2x2 or 4x4 matrices, got size {size}.")
    values = [compute(r) for r in states.rho]
    return Observations(states.times, {
        name: np.array([v[name] for v in values]) for name in names})


def _per_state(
    name: str, states: States, measure: Callable[[np.ndarray], float]
) -> Observations:
    """Observes one scalar measure of each density matrix."""
    _check_states(states)
    return Observations(
        states.times, {name: np.array([measure(r) for r in states.rho])})


def coherence_l1(states: States) -> Observations:
    """Observes the l1 coherence in the computational basis.

    Args:
        states: The states.

    Returns:
        The series ``coherence_l1``.

    Raises:
        ValueError: If `states` is empty.
    """
    return _per_state("coherence_l1", states, measures.coherence_l1)


def purity(states: States) -> Observations:
    """Observes the purity Tr(rho^2).

    Args:
        states: The states.

    Returns:
        The series ``purity``.

    Raises:
        ValueError: If `states` is empty.
    """
    return _per_state("purity", states, measures.purity)


def concurrence(states: States) -> Observations:
    """Observes the Wootters concurrence of two qubits.

    Args:
        states: Two-qubit states.

    Returns:
        The series ``concurrence``.

    Raises:
        ValueError: If `states` is empty or not 4x4.
    """
    _check_states(states, 4)
    return _per_state("concurrence", states, measures.concurrence)


def correlators(states: States) -> Observations:
    """Observes <sigma_a x sigma_b> of two qubits for all a, b (R23).

    Args:
        states: Two-qubit states.

    Returns:
        The nine series ``corr_xx`` ... ``corr_zz``.

    Raises:
        ValueError: If `states` is empty or not 4x4.
    """
    _check_states(states, 4)
    rhos = np.asarray(states.rho)
    return Observations(states.times, {
        name: np.real(np.einsum("tij,ji->t", rhos, op))
        for name, op in CORR.items()
    })


def combine(*observations: Observations) -> Observations:
    """Merges observations of one trajectory, keeping the argument order.

    Args:
        *observations: Observations on the same time grid.

    Returns:
        One `Observations` with all series.

    Raises:
        ValueError: If none is given, the time grids differ, or a name
            occurs twice.
    """
    if not observations:
        raise ValueError("combine needs at least one Observations.")
    times = observations[0].times
    series: dict[str, np.ndarray] = {}
    for obs in observations:
        if not np.array_equal(obs.times, times):
            raise ValueError("Observations must share one time grid.")
        repeated = sorted(set(series) & set(obs.series))
        if repeated:
            raise ValueError(f"Duplicate series names: {repeated}.")
        series.update(obs.series)
    return Observations(times, series)


def select(obs: Observations, names: Iterable[str]) -> Observations:
    """Keeps the named series, in the order of ``names``.

    Args:
        obs: The observations.
        names: Names to keep.

    Returns:
        The selected series.

    Raises:
        TypeError: If `names` is a single string.
        ValueError: If a name is not a series of `obs`, or none is kept.
    """
    if isinstance(names, str):
        raise TypeError(
            f"Expected a collection of names, got the string {names!r}.")
    names = list(names)
    missing = [n for n in names if n not in obs.series]
    if missing:
        raise ValueError(
            f"Unknown series {missing}; available: {list(obs.series)}.")
    if not names:
        raise ValueError("select needs at least one name.")
    return Observations(obs.times, {n: obs.series[n] for n in names})


def _noisy_names(
    obs: Observations, sigma: float, rng: object,
    names: Sequence[str] | None,
) -> list[str]:
    """Validates the noise arguments; returns the names to perturb."""
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError(
            f"sigma must be finite and non-negative, got {sigma!r}.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {rng!r}.")
    if names is None:
        return list(obs.series)
    unknown = [n for n in names if n not in obs.series]
    if unknown:
        raise ValueError(
            f"Unknown series {unknown}; available: {list(obs.series)}.")
    return [n for n in obs.series if n in set(names)]


def _perturbed(
    obs: Observations, targets: list[str], noise: np.ndarray
) -> Observations:
    """Adds noise of shape (n_times, len(targets)) to the target series."""
    series = dict(obs.series)
    for k, name in enumerate(targets):
        series[name] = obs.series[name] + noise[:, k]
    return Observations(obs.times, series)


def gaussian_noise(
    obs: Observations,
    sigma: float,
    rng: np.random.Generator,
    names: Sequence[str] | None = None,
) -> Observations:
    """Adds independent Gaussian noise to every entry of the series.

    Args:
        obs: Exact observations.
        sigma: Noise standard deviation; 0 returns `obs` and draws nothing.
        rng: Source of randomness. Draws one normal array of shape
            (n_times, n_noisy_series), series in the order of `obs`.
        names: Series to perturb; None perturbs all.

    Returns:
        Noisy observations; unlisted series are unchanged.

    Raises:
        ValueError: If `sigma` is negative or not finite, or a name is
            unknown.
        TypeError: If `rng` is not a Generator.
    """
    targets = _noisy_names(obs, sigma, rng, names)
    if sigma == 0.0:
        return obs
    noise = rng.normal(0.0, sigma, size=(len(obs.times), len(targets)))
    return _perturbed(obs, targets, noise)


def ar1_noise(
    obs: Observations,
    sigma: float,
    phi: float,
    rng: np.random.Generator,
    names: Sequence[str] | None = None,
) -> Observations:
    """Adds AR(1) noise along time to the series.

    The unit-variance noise is drawn from a standard normal, shape
    (n_times, n_noisy_series), and filtered with `ar1_filter`, so equal
    generator states give the same realisation for every `sigma` and `phi`.

    Args:
        obs: Exact observations.
        sigma: Noise standard deviation.
        phi: Lag-one correlation, in (-1, 1); 0 gives white noise.
        rng: Source of randomness.
        names: Series to perturb; None perturbs all.

    Returns:
        Noisy observations; unlisted series are unchanged.

    Raises:
        ValueError: If `sigma` is negative or not finite, `phi` is not in
            (-1, 1), or a name is unknown.
        TypeError: If `rng` is not a Generator.
    """
    targets = _noisy_names(obs, sigma, rng, names)
    eta = rng.normal(0.0, 1.0, size=(1, len(obs.times), len(targets)))
    return _perturbed(obs, targets, sigma * ar1_filter(eta, phi)[0])
