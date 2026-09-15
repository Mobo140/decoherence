"""OLS physics prior for the decoherence-time estimate.

Extracted verbatim from lstm_predictor.py (task 19) -- no behaviour change.
The coherence decays as C(t) = C0*exp(-gamma*t/2), so log C is linear in t and
the OLS slope gives T2 = 2/gamma = -1/slope. See lstm_predictor.py for the
full derivation.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Physics estimate
# ---------------------------------------------------------------------------

def _n_physics_from_state(state_dict: dict, hidden_size: int) -> int | None:
    """Infer how many physics scalars the checkpoint concatenated (4 / 7 / 8)."""
    weight = state_dict.get("regression_head.0.weight")
    if weight is None:
        return None
    n = int(weight.shape[1] - hidden_size * 2)
    if n < 1:
        return None
    return n


_ENVELOPE_RAW_R2 = 0.7


def _ols_log_slope(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """OLS slope and R² of log(series) vs time. series: (T,) or (N, T)."""
    x = np.asarray(series, dtype=np.float64)
    batched = x.ndim == 2
    if not batched:
        x = x[np.newaxis, :]
    log_c = np.log(np.maximum(x, 1e-8))
    T = log_c.shape[1]
    t_arr = np.arange(T, dtype=np.float64) * dt
    t_c = t_arr - t_arr.mean()
    denom = float(np.dot(t_c, t_c)) + 1e-12
    lc_mean = log_c.mean(axis=1, keepdims=True)
    slopes = ((log_c - lc_mean) @ t_c) / denom
    fitted = lc_mean + slopes[:, np.newaxis] * t_c
    ss_res = np.sum((log_c - fitted) ** 2, axis=1)
    ss_tot = np.sum((log_c - lc_mean) ** 2, axis=1) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    if not batched:
        return slopes[0], r2[0]
    return slopes, r2


def _coherence_envelope(coh: np.ndarray) -> np.ndarray:
    """Slow decay envelope of L1 coherence. coh: (T,) or (N, T).

    L1 is nonnegative, so Hilbert on raw C(t) distorts a decaying carrier.
    Prefer the upper peak envelope when there are enough maxima (TFIM-like);
    otherwise fall back to a reflect-padded Hilbert envelope.
    """
    from scipy.signal import find_peaks, hilbert

    x = np.asarray(coh, dtype=np.float64)
    batched = x.ndim == 2
    if not batched:
        x = x[np.newaxis, :]
    out = np.empty_like(x)
    t = x.shape[1]
    idx = np.arange(t)
    for i, row in enumerate(x):
        peaks, _ = find_peaks(row, distance=2)
        extra = []
        if row[0] >= row[1]:
            extra.append(0)
        if row[-1] >= row[-2]:
            extra.append(t - 1)
        if extra:
            peaks = np.unique(np.concatenate([peaks, np.array(extra, dtype=int)]))
        if peaks.size >= 3:
            out[i] = np.interp(idx, peaks.astype(np.float64), row[peaks])
            continue
        padded = np.concatenate([row[::-1], row, row[::-1]])
        out[i] = np.abs(hilbert(padded))[t : 2 * t]
    out = np.maximum(out, 1e-8)
    return out[0] if not batched else out


def _maybe_envelope_slope(coh: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Use Hilbert envelope OLS when raw log(C) is not linear (2q oscillations)."""
    slopes, r2 = _ols_log_slope(coh, dt)
    env_slopes, env_r2 = _ols_log_slope(_coherence_envelope(coh), dt)
    use_env = (r2 < _ENVELOPE_RAW_R2) & (env_r2 > r2)
    if np.ndim(slopes) == 0:
        if use_env:
            return env_slopes, env_r2
        return slopes, r2
    return np.where(use_env, env_slopes, slopes), np.where(use_env, env_r2, r2)


def physics_remaining(
    window: np.ndarray,
    t_obs: float,
    dt: float = 0.1,
    t_max: float = 20.0,
    adaptive_r2_threshold: float = 0.0,
) -> Tuple[float, float, float]:
    """Estimate remaining decoherence time from the OLS slope of log(coherence_l1).

    Returns (remaining_estimate, slope, ols_r2).

    On oscillatory windows (raw OLS R² < 0.7) the slope is taken from the
    peak / Hilbert envelope of C(t) when that fit is straighter — same
    −1/slope identity, but on the slow decay mode instead of the carrier.

    ``adaptive_r2_threshold`` (0 = disabled): when the chosen OLS R² is
    below this threshold the physics estimate is set to 0 so the LSTM
    residual takes over.  A value of 0.7 works well for 2-qubit systems.
    """
    coh = np.abs(window[:, -1]) + 1e-8
    slope, ols_r2 = _maybe_envelope_slope(coh, dt)
    slope = float(slope)
    ols_r2 = float(ols_r2)

    if slope >= -1e-6 or (adaptive_r2_threshold > 0.0 and ols_r2 < adaptive_r2_threshold):
        return 0.0, slope, ols_r2

    raw_est = -1.0 / slope - t_obs
    remaining_est = float(np.clip(raw_est, 0.0, t_max))
    return remaining_est, slope, ols_r2
