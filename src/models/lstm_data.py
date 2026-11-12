"""Arrays, batches and tensors for `LSTMPredictor`."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch.utils import data as torch_data

from src.data import dataset as dataset_lib
from src.models import physics_prior
from src.training import standardize


class Arrays(NamedTuple):
    """The per-window arrays of one split, in the order of a batch.

    The field order is the order of the tensors in a batch from
    `make_loader`; the training step unpacks batches by it.

    Attributes:
        windows: Windows with log columns, shape (N, T, F + 2).
        residual: Regression target (remaining time, minus the physics
            estimate for a residual network), shape (N,).
        t_obs: Observation time.
        slope: OLS slope of log coherence.
        log_coh: Log coherence at t_obs.
        phys: Physics estimate of the remaining time.
        r2: OLS R^2.
        int_code: Interaction type code.
        dis_code: Dissipator type code.
        j: Coupling J.
        risk: Binary risk label.
        remaining: True remaining time.
        censored: 1.0 where the remaining time is only a lower bound.
    """

    windows: np.ndarray
    residual: np.ndarray
    t_obs: np.ndarray
    slope: np.ndarray
    log_coh: np.ndarray
    phys: np.ndarray
    r2: np.ndarray
    int_code: np.ndarray
    dis_code: np.ndarray
    j: np.ndarray
    risk: np.ndarray
    remaining: np.ndarray
    censored: np.ndarray


def add_log(windows: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Appends log coherence_l1 and log purity as two extra channels.

    Args:
        windows: Observables, shape (..., F).
        feature_names: Names of the F channels. A missing `coherence_l1`
            defaults to the last channel and a missing `purity` to the one
            before it.

    Returns:
        float32 array of shape (..., F + 2).
    """
    names = list(feature_names)
    ci = names.index("coherence_l1") if "coherence_l1" in names \
        else windows.shape[-1] - 1
    pi = names.index("purity") if "purity" in names else windows.shape[-1] - 2
    lc = np.log(np.abs(windows[..., ci:ci + 1]) + 1e-8).astype(np.float32)
    lp = np.log(np.abs(windows[..., pi:pi + 1]) + 1e-8).astype(np.float32)
    return np.concatenate([windows.astype(np.float32), lc, lp], axis=-1)


def scalar_tensor(
    value: float, scaler: standardize.Standardizer, device: str
) -> torch.Tensor:
    """Standardises one scalar into a tensor of shape (1,) on `device`."""
    return torch.tensor([scaler.apply(value)], dtype=torch.float32).to(device)


def _context_column(
    dataset: dataset_lib.Dataset, name: str, idx: np.ndarray
) -> np.ndarray:
    """Per-window context array, or zeros if the dataset does not have it."""
    column = getattr(dataset, name)
    if len(column) == len(dataset.sequences):
        return column[idx]
    return np.zeros(len(idx), dtype=np.float32)


def prepare_arrays(
    dataset: dataset_lib.Dataset,
    idx: np.ndarray,
    *,
    dt: float,
    t_max: float,
    adaptive_r2_threshold: float,
    use_physics_prior: bool,
    residual_output: bool,
) -> Arrays:
    """Computes the physics scalars and targets for the rows `idx`.

    Args:
        dataset: Source dataset.
        idx: Row indices of the split.
        dt: Time step between rows of a window.
        t_max: Upper bound on the physics estimate.
        adaptive_r2_threshold: See `physics_prior.batch_physics`.
        use_physics_prior: If False the slope and physics estimate are
            zeroed, so the network sees only the raw windows.
        residual_output: If True the target is the remaining time minus the
            physics estimate, else the remaining time itself.

    Returns:
        The arrays of the split; windows are not yet standardised.
    """
    windows = dataset.sequences[idx]
    remaining = dataset.remaining_time[idx]
    t_obs = dataset.t_obs[idx]
    risk = dataset.risk_labels[idx]
    slopes, log_coh, phys, r2 = physics_prior.batch_physics(
        windows, dataset.feature_names, dt, t_obs, t_max,
        adaptive_r2_threshold=adaptive_r2_threshold,
    )
    if not use_physics_prior:
        phys = np.zeros_like(phys)
        slopes = np.zeros_like(slopes)
    target = remaining - phys if residual_output else remaining
    if len(dataset.censored) == len(dataset.sequences):
        censored = dataset.censored[idx].astype(np.float32)
    else:
        censored = np.zeros(len(idx), dtype=np.float32)
    return Arrays(
        windows=add_log(windows, dataset.feature_names),
        residual=target.astype(np.float32),
        t_obs=t_obs,
        slope=slopes,
        log_coh=log_coh,
        phys=phys,
        r2=r2,
        int_code=_context_column(dataset, "interaction_type_codes", idx),
        dis_code=_context_column(dataset, "dissipator_type_codes", idx),
        j=_context_column(dataset, "J_values", idx),
        risk=risk,
        remaining=remaining,
        censored=censored,
    )


def make_loader(
    arrays: Arrays, batch_size: int, shuffle: bool
) -> torch_data.DataLoader:
    """Wraps the arrays of a split into a DataLoader of float32 batches."""
    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
    return torch_data.DataLoader(
        torch_data.TensorDataset(*tensors),
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
    )
