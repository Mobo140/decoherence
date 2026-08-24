"""Hilbert envelope prior: keep 1-exp formula, switch on oscillations."""
from __future__ import annotations

import numpy as np

from src.infrastructure.ml.lstm_predictor import (
    _ols_log_slope,
    physics_remaining,
)


def _window(coh: np.ndarray) -> np.ndarray:
    extra = np.zeros((len(coh), 2), dtype=np.float64)
    extra[:, -1] = coh
    return extra


def test_pure_exponential_keeps_one_over_slope():
    dt, lam, t_obs = 0.1, 0.25, 3.0
    t = np.arange(30) * dt
    coh = np.exp(-lam * t)
    rem, slope, r2 = physics_remaining(_window(coh), t_obs, dt=dt, t_max=20.0)
    assert r2 > 0.99
    assert abs(slope + lam) < 0.02
    assert abs(rem - (1.0 / lam - t_obs)) < 0.3


def test_oscillation_envelope_beats_raw_ols():
    dt, lam, t_obs = 0.1, 0.25, 3.0
    t = np.arange(50) * dt
    true_rem = 1.0 / lam - t_obs
    coh = np.exp(-lam * t) * (0.12 + 0.88 * np.abs(np.cos(4.2 * t)))
    raw_slope, raw_r2 = _ols_log_slope(coh, dt)
    rem, slope, r2 = physics_remaining(_window(coh), t_obs, dt=dt, t_max=20.0)
    raw_rem = float(np.clip(-1.0 / raw_slope - t_obs, 0.0, 20.0)) if raw_slope < -1e-6 else 20.0
    assert r2 > raw_r2
    assert abs(rem - true_rem) < abs(raw_rem - true_rem)
    assert abs(rem - true_rem) < 1.5
