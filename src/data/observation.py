"""Observation models: how exact window averages become model inputs.

Simulation gives exact expectation values. An observation model turns a
dataset of exact windows into the windows a measurement would produce.
`OBSERVATIONS` maps an alias to such a function, all with the signature

    observe(dataset, *, sigma, phi=0.0, rng) -> Dataset

where `sigma` is the noise standard deviation, `phi` the lag-one
correlation of the noise and `rng` the source of randomness. The input
dataset is never modified. `noise_augmented` builds the larger training
sets of the noise-aware experiments (R16, R21, R22).
"""
from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Mapping

import numpy as np

from src.data import dataset as dataset_lib

# Per-window arrays that noise_augmented repeats for every noisy copy.
_PER_WINDOW_FIELDS = (
    "sequences", "t_obs", "t_decoh_abs", "remaining_time", "risk_labels",
    "interaction_type_codes", "dissipator_type_codes", "J_values",
    "censored",
)

# Range of the noise level (log-uniform) used for training augmentation.
_AUGMENT_SIGMA_RANGE = (0.005, 0.12)


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
    """
    phi = np.asarray(phi, dtype=np.float64)
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


def exact(
    dataset: dataset_lib.Dataset,
    *,
    sigma: float = 0.0,
    phi: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dataset_lib.Dataset:
    """Observes the exact expectation values, without noise.

    Args:
        dataset: Dataset of exact windows.
        sigma: Must be 0.
        phi: Must be 0.
        rng: Unused.

    Returns:
        `dataset` itself.

    Raises:
        ValueError: If `sigma` or `phi` is not 0.
    """
    if sigma != 0.0 or phi != 0.0:
        raise ValueError("The exact observation takes no noise")
    return dataset


def gaussian(
    dataset: dataset_lib.Dataset,
    *,
    sigma: float,
    phi: float = 0.0,
    rng: np.random.Generator,
) -> dataset_lib.Dataset:
    """Adds independent Gaussian noise to every entry of every window.

    Args:
        dataset: Dataset of exact windows.
        sigma: Noise standard deviation; 0 returns `dataset` itself.
        phi: Must be 0, the noise is white.
        rng: Source of randomness. Draws one float64 normal array of the
            shape of `dataset.sequences`.

    Returns:
        A dataset with noisy windows.

    Raises:
        ValueError: If `phi` is not 0.
    """
    if phi != 0.0:
        raise ValueError("Gaussian noise is white; use 'ar1' for phi > 0")
    if sigma == 0.0:
        return dataset
    noise = rng.normal(0.0, sigma, size=dataset.sequences.shape)
    return dataset.with_sequences(dataset.sequences + noise.astype(np.float32))


def ar1(
    dataset: dataset_lib.Dataset,
    *,
    sigma: float,
    phi: float,
    rng: np.random.Generator,
) -> dataset_lib.Dataset:
    """Adds AR(1) noise along the time axis of every window.

    The unit-variance noise is drawn from a standard normal and filtered
    with `ar1_filter`, so equal generator states give the same realisation
    for every `sigma` and `phi`.

    Args:
        dataset: Dataset of exact windows.
        sigma: Noise standard deviation.
        phi: Lag-one correlation of the noise, 0 for white noise.
        rng: Source of randomness. Draws one float64 normal array of the
            shape of `dataset.sequences`.

    Returns:
        A dataset with noisy windows.
    """
    eta = rng.normal(0.0, 1.0, size=dataset.sequences.shape)
    noise = ar1_filter(eta, phi)
    return dataset.with_sequences(dataset.sequences + sigma * noise)


Observation = Callable[..., dataset_lib.Dataset]

OBSERVATIONS: Mapping[str, Observation] = types.MappingProxyType({
    "exact": exact,
    "gaussian": gaussian,
    "ar1": ar1,
})


def noise_augmented(
    dataset: dataset_lib.Dataset,
    copies: int,
    rng: np.random.Generator,
    phi_max: float = 0.0,
) -> dataset_lib.Dataset:
    """Appends noisy copies of every train and val window.

    The noise goes into the raw window, so every quantity a model derives
    from it (log channels, slope, OLS R^2, T_phys) comes from the noisy
    copy. Test windows are left untouched. Each copy draws, in this order,
    one log-uniform sigma per window, a normal array and, if
    `phi_max > 0`, one correlation per window.

    Args:
        dataset: Dataset of exact windows.
        copies: Number of noisy copies of the train and val windows.
        rng: Source of randomness.
        phi_max: If positive, the noise is AR(1) in time with a lag-one
            correlation drawn per window from U[0, phi_max]; if 0, the
            noise is white.

    Returns:
        A dataset with the original windows followed by the copies. The
        copies join the train and val splits of their originals.
    """
    n = len(dataset.sequences)
    source = np.concatenate([dataset.train_idx, dataset.val_idx])
    is_train = np.concatenate([np.ones(len(dataset.train_idx), bool),
                               np.zeros(len(dataset.val_idx), bool)])
    parts = {k: [getattr(dataset, k)] for k in _PER_WINDOW_FIELDS}
    new_train, new_val = [dataset.train_idx], [dataset.val_idx]
    offset = n
    log_lo = np.log(_AUGMENT_SIGMA_RANGE[0])
    log_hi = np.log(_AUGMENT_SIGMA_RANGE[1])
    for _ in range(copies):
        sig = np.exp(rng.uniform(log_lo, log_hi, len(source))).astype(
            np.float32)
        eta = rng.normal(
            0.0, 1.0, dataset.sequences[source].shape).astype(np.float32)
        if phi_max > 0.0:
            eta = ar1_filter(eta, rng.uniform(0.0, phi_max, len(source)))
        noisy = dataset.sequences[source] + eta * sig[:, None, None]
        for k in _PER_WINDOW_FIELDS:
            arr = getattr(dataset, k)
            if k == "sequences":
                parts[k].append(noisy)
            else:
                parts[k].append(arr[source] if len(arr) == n else arr)
        new_idx = offset + np.arange(len(source))
        new_train.append(new_idx[is_train])
        new_val.append(new_idx[~is_train])
        offset += len(source)
    fields = {
        k: np.concatenate(v) if len(getattr(dataset, k)) == n
        else getattr(dataset, k)
        for k, v in parts.items()
    }
    return dataclasses.replace(
        dataset, **fields,
        train_idx=np.concatenate(new_train),
        val_idx=np.concatenate(new_val))
