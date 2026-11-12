"""Observation: named series computed from the states of a trajectory.

An observation function maps ``States`` to ``Observations``, a set of named
time series. Exact observations come straight from the density matrices;
the noise functions act on a trajectory's series, so a model sees what a
measurement of that one trajectory would give. ``combine`` and ``select``
assemble the series a model receives, addressed by name, never by column
position. The formulas live in ``src.physics``; this module only collects
them into series. ``MODEL_TAIL`` and ``model_layout`` fix the one column
order the models still rely on.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.blocks.evolve import States
from src.physics import measures, observables, paulis

# Two-body Pauli correlators <sigma_a x sigma_b>, keyed by series name.
CORR = {
    f"corr_{a}{b}": np.kron(paulis.PAULI[a], paulis.PAULI[b])
    for a in "xyz" for b in "xyz"
}


# The series every model expects as its last two input columns, in this
# order: the models read them by position.
MODEL_TAIL = ("purity", "coherence_l1")


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


SETTINGS = ("local", "tomography")
_AXES = "xyz"
# Eigenvalues of the two qubits for the outcomes (++, +-, -+, --).
_SIGN_1 = np.array([1.0, 1.0, -1.0, -1.0])
_SIGN_2 = np.array([1.0, -1.0, 1.0, -1.0])


def _eigenprojectors(axis: str) -> list[np.ndarray]:
    """Returns the projectors on the +1 and -1 eigenstates of sigma_axis."""
    sigma = paulis.PAULI[axis]
    return [(paulis.IDENTITY + sigma) / 2, (paulis.IDENTITY - sigma) / 2]


def _outcome_probabilities(
    rhos: np.ndarray, projectors: list[np.ndarray]
) -> np.ndarray:
    """Born probabilities of each outcome at each time, shape (T, K).

    Rounding can leave a probability a hair below zero; those are set to
    zero and each row is renormalised.
    """
    p = np.stack([
        np.real(np.einsum("tij,ji->t", rhos, projector))
        for projector in projectors], axis=1)
    p = np.clip(p, 0.0, None)
    total = p.sum(axis=1, keepdims=True)
    if not np.all(total > 0):
        raise ValueError("A density matrix gives no measurement outcome.")
    return p / total


def _draw_counts(
    rng: np.random.Generator, n_shots: int, p: np.ndarray
) -> np.ndarray:
    """Draws the outcome counts of n_shots shots for each time."""
    return np.stack([rng.multinomial(n_shots, row) for row in p])


def finite_shots(
    states: States,
    n_shots: int,
    settings: str,
    rng: np.random.Generator,
) -> Observations:
    """Observes the Pauli means from a finite number of projective shots.

    For every measurement basis and every time the outcome probabilities
    are computed from rho(t) and ``n_shots`` outcomes are drawn from a
    multinomial; the means are the sample averages of the eigenvalues, so
    each is unbiased with variance (1 - <sigma>^2) / n_shots.

    ``"local"`` measures both qubits in the same basis x, y and z (3
    settings) and gives <sigma_a> of each qubit, the series of
    `pauli_means`. ``"tomography"`` measures all 3^n product bases. For two
    qubits it adds the correlators <sigma_a sigma_b> (the series of
    `correlators`), and it reconstructs rho by linear inversion, which
    gives ``purity`` and ``coherence_l1`` of the estimate. For one qubit
    local and tomography are the same measurement and always give both.
    Two qubits measured locally have no purity or coherence series: they
    need tomography. The single-qubit means of two qubits come from the
    basis in which both qubits are measured along that axis.

    Draws are made setting by setting, in the order xx, xy, ... (tomography)
    or x, y, z (local), and within a setting time by time.

    Args:
        states: States of one or two qubits.
        n_shots: Shots per basis and time, at least 1.
        settings: ``"local"`` or ``"tomography"``.
        rng: Source of randomness.

    Returns:
        Series named as the exact blocks; see above for which exist.

    Raises:
        ValueError: If `n_shots` is not a positive integer, `settings` is
            unknown, `states` is empty or not 2x2 or 4x4, or a density
            matrix gives no outcome.
        TypeError: If `rng` is not a Generator.
    """
    if int(n_shots) != n_shots or n_shots < 1:
        raise ValueError(
            f"n_shots must be a positive integer, got {n_shots!r}.")
    if settings not in SETTINGS:
        raise ValueError(
            f"settings must be one of {SETTINGS}, got {settings!r}.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {rng!r}.")
    _check_states(states)
    size = np.shape(states.rho[0])[0]
    if size not in (2, 4):
        raise ValueError(
            f"finite_shots needs 2x2 or 4x4 matrices, got size {size}.")
    rhos = np.asarray(states.rho)
    n_shots = int(n_shots)
    if size == 2:
        means = {}
        for axis in _AXES:
            p = _outcome_probabilities(rhos, _eigenprojectors(axis))
            counts = _draw_counts(rng, n_shots, p)
            means[axis] = (counts[:, 0] - counts[:, 1]) / n_shots
        series = {f"sigma_{a}": means[a] for a in _AXES}
        return Observations(
            states.times, {**series, **_reconstruction_series(
                [means[a] for a in _AXES], None)})

    pairs = ([(a, a) for a in _AXES] if settings == "local"
             else [(a, b) for a in _AXES for b in _AXES])
    one, two, corr = {}, {}, {}
    for a, b in pairs:
        projectors = [
            np.kron(pa, pb) for pa in _eigenprojectors(a)
            for pb in _eigenprojectors(b)]
        counts = _draw_counts(
            rng, n_shots, _outcome_probabilities(rhos, projectors))
        if a == b:
            one[a] = counts @ _SIGN_1 / n_shots
            two[a] = counts @ _SIGN_2 / n_shots
        corr[f"corr_{a}{b}"] = counts @ (_SIGN_1 * _SIGN_2) / n_shots
    series = {f"sigma_{a}1": one[a] for a in _AXES}
    series.update({f"sigma_{a}2": two[a] for a in _AXES})
    if settings == "local":
        return Observations(states.times, series)
    series.update({name: corr[name] for name in CORR})
    series.update(_reconstruction_series(
        [one[a] for a in _AXES], ([two[a] for a in _AXES], corr)))
    return Observations(states.times, series)


def _reconstruction_series(
    first: list[np.ndarray], second: tuple | None
) -> dict[str, np.ndarray]:
    """Purity and l1 coherence of the linear-inversion estimate of rho.

    Args:
        first: Estimated <sigma_a> of the first (or only) qubit, a = x, y, z.
        second: None for one qubit; otherwise the means of the second qubit
            and the dict of correlators ``corr_ab``.
    """
    n_times = len(first[0])
    estimates = []
    for t in range(n_times):
        rho = paulis.IDENTITY.copy()
        if second is None:
            rho = (rho + sum(
                first[k][t] * paulis.PAULI[a] for k, a in enumerate(_AXES))
            ) / 2
        else:
            means_2, corr = second
            rho = np.kron(paulis.IDENTITY, paulis.IDENTITY)
            for k, a in enumerate(_AXES):
                rho = rho + first[k][t] * np.kron(
                    paulis.PAULI[a], paulis.IDENTITY)
                rho = rho + means_2[k][t] * np.kron(
                    paulis.IDENTITY, paulis.PAULI[a])
                for b in _AXES:
                    rho = rho + corr[f"corr_{a}{b}"][t] * np.kron(
                        paulis.PAULI[a], paulis.PAULI[b])
            rho = rho / 4
        estimates.append(rho)
    return {
        "purity": np.array([measures.purity(r) for r in estimates]),
        "coherence_l1": np.array(
            [measures.coherence_l1(r) for r in estimates]),
    }


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


def ar1_filter(eta: np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    """Turns white noise into stationary unit-variance AR(1) noise.

    The filter runs along axis 1:

        e_0 = eta_0,   e_t = phi e_{t-1} + sqrt(1 - phi^2) eta_t

    Args:
        eta: Unit-variance white noise, shape (n_windows, n_steps, ...).
        phi: Lag-one correlation, a scalar or one value per window. With
            phi = 0 `eta` is returned unchanged, so white and correlated
            noise can share one realisation of `eta`.

    Returns:
        Noise of the same shape and dtype as `eta`.

    Raises:
        ValueError: If `eta` is not at least 2-D with one step or more,
            `phi` is neither a scalar nor a 1-D array with one value per
            window, or `phi` has an absolute value of 1 or more (or is
            NaN), for which the noise is not stationary.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if eta.ndim < 2 or eta.shape[1] == 0:
        raise ValueError(
            "eta must have shape (n_windows, n_steps, ...) with at least "
            f"one step, got {eta.shape}.")
    if phi.ndim > 1 or (phi.ndim == 1 and phi.shape[0] != eta.shape[0]):
        raise ValueError(
            f"phi must be a scalar or have one value per window "
            f"({eta.shape[0]}), got shape {phi.shape}.")
    if not np.all(np.abs(phi) < 1.0):
        raise ValueError("phi must satisfy |phi| < 1.")
    if not phi.any():
        return eta
    if phi.ndim == 1:
        phi = phi.reshape((-1,) + (1,) * (eta.ndim - 2))
    c = np.sqrt(1.0 - phi * phi)
    e = np.empty(eta.shape, dtype=np.float64)
    e[:, 0] = eta[:, 0]
    for t in range(1, eta.shape[1]):
        e[:, t] = phi * e[:, t - 1] + c * eta[:, t]
    return e.astype(eta.dtype)


def model_layout(obs: Observations) -> Observations:
    """Orders the series as the models read them.

    The series of ``MODEL_TAIL`` move to the end, in that order; the others
    keep their order.

    Args:
        obs: The observations.

    Returns:
        The reordered observations.

    Raises:
        ValueError: If a series of `MODEL_TAIL` is missing.
    """
    missing = [n for n in MODEL_TAIL if n not in obs.series]
    if missing:
        raise ValueError(
            f"model_layout needs the series {missing}; "
            f"available: {list(obs.series)}. Two qubits measured with "
            "finite_shots(settings='local') have no purity or "
            "coherence_l1: they need tomography.")
    head = [n for n in obs.series if n not in MODEL_TAIL]
    return select(obs, head + list(MODEL_TAIL))


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
