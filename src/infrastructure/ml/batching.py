"""Batch construction helpers for LSTMPredictor training.

Extracted verbatim from lstm_predictor.py (task 19) -- no behaviour change.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .physics_prior import _maybe_envelope_slope



# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _batch_physics(
    X: np.ndarray,
    feature_names: list,
    dt: float,
    t_obs: np.ndarray,
    t_max: float = 20.0,
    adaptive_r2_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (slopes, log_cohs, physics_remaining, ols_r2) for (N, T, F) array.

    Returns a 4-tuple; ols_r2 is the OLS R² of log(C) vs t for each sample.
    High R² → log-linear decay (XXZ-like); low R² → oscillatory (TFIM-like).

    When ``adaptive_r2_threshold`` > 0, samples where OLS R²(log C vs t) is
    below the threshold get phys_rem = 0 — the LSTM residual head takes full
    responsibility.  This prevents the physics prior from misleading the model
    on oscillatory two-qubit coherence (e.g. TFIM).
    """
    names = list(feature_names)
    ci = names.index("coherence_l1") if "coherence_l1" in names else X.shape[-1] - 1

    coh = np.abs(X[:, :, ci]) + 1e-8
    log_coh_at_obs = np.log(coh[:, -1])
    slopes, ols_r2 = _maybe_envelope_slope(coh, dt)

    if adaptive_r2_threshold > 0.0:
        bad_prior = (slopes >= -1e-6) | (ols_r2 < adaptive_r2_threshold)
    else:
        bad_prior = slopes >= -1e-6

    raw_rem = np.where(~bad_prior, -1.0 / np.where(bad_prior, -1.0, slopes) - t_obs, 0.0)
    phys_rem = np.clip(raw_rem, 0.0, t_max)

    return (slopes.astype(np.float32), log_coh_at_obs.astype(np.float32),
            phys_rem.astype(np.float32), ols_r2.astype(np.float32))


def _add_log(X: np.ndarray, feature_names: list) -> np.ndarray:
    names = list(feature_names)
    ci = names.index("coherence_l1") if "coherence_l1" in names else X.shape[-1] - 1
    pi = names.index("purity")       if "purity"       in names else X.shape[-1] - 2
    lc = np.log(np.abs(X[..., ci:ci+1]) + 1e-8).astype(np.float32)
    lp = np.log(np.abs(X[..., pi:pi+1]) + 1e-8).astype(np.float32)
    return np.concatenate([X.astype(np.float32), lc, lp], axis=-1)


def _ns(val: float, mean: float, std: float, device: str) -> torch.Tensor:
    return torch.tensor([(val - mean) / std], dtype=torch.float32).to(device)


def _loader(X, y, t, s, lc, ph, r2, ic, dc, j, r, rem, cens, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X,    dtype=torch.float32),
        torch.tensor(y,    dtype=torch.float32),
        torch.tensor(t,    dtype=torch.float32),
        torch.tensor(s,    dtype=torch.float32),
        torch.tensor(lc,   dtype=torch.float32),
        torch.tensor(ph,   dtype=torch.float32),
        torch.tensor(r2,   dtype=torch.float32),
        torch.tensor(ic,   dtype=torch.float32),
        torch.tensor(dc,   dtype=torch.float32),
        torch.tensor(j,    dtype=torch.float32),
        torch.tensor(r,    dtype=torch.float32),
        torch.tensor(rem,  dtype=torch.float32),
        torch.tensor(cens, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
