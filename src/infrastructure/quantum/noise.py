"""Correlated (AR(1)) measurement-noise helper used by the noise experiments."""
from __future__ import annotations

import numpy as np


def ar1_filter(eta: np.ndarray, phi) -> np.ndarray:
    """Turn white noise into stationary unit-variance AR(1) noise along axis 1.

        e_0 = eta_0,   e_t = phi e_{t-1} + sqrt(1 - phi^2) eta_t

    ``eta`` has shape (n_windows, n_steps, ...) with unit variance; ``phi``
    is a scalar or one value per window. With phi = 0 the input is returned
    unchanged, so white and correlated noise can share one realisation of
    ``eta`` (R21, R22).
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
