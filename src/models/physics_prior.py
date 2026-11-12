"""OLS physics prior for the decoherence-time estimate.

The coherence decays as C(t) = C0*exp(-gamma*t/2), so log C is linear in t
and the OLS slope gives T2 = 2/gamma = -1/slope. For oscillatory two-qubit
coherence the slope is taken from the slow envelope of C(t) instead.

Every model that uses the prior (`physics_only`, `lstm`) goes through
`physics_remaining` for one window and `batch_physics` for a whole array of
windows; `remaining_to_risk` turns a remaining time into the risk score
that all predictors return.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

# Below this R^2 of the raw log-linear fit the window is considered
# oscillatory and the envelope fit is tried instead.
ENVELOPE_RAW_R2 = 0.7

# A slope this close to zero (or positive) means no decay was observed.
_FLAT_SLOPE = -1e-6


def remaining_to_risk(
    remaining: float, horizon: float, k: float = 5.0
) -> float:
    """Maps a remaining time to P(remaining <= horizon).

    The same logistic is used by every predictor: risk =
    sigmoid(-k * (remaining - horizon)). Network risk heads are only a
    training auxiliary; their logits are not used at prediction time.

    Args:
        remaining: Predicted time until decoherence.
        horizon: Alarm horizon.
        k: Sharpness of the logistic.

    Returns:
        Risk score in [0, 1].
    """
    z = np.clip(k * (remaining - horizon), -50.0, 50.0)
    return float(1.0 / (1.0 + np.exp(z)))


def ols_log_slope(
    series: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fits log(series) = a + slope * t by least squares.

    Args:
        series: Positive values, shape (T,) or (N, T).
        dt: Time step between samples.

    Returns:
        (slope, r2) with the shape of `series` without its last axis
        (scalars for a 1-D input).
    """
    x = np.asarray(series, dtype=np.float64)
    batched = x.ndim == 2
    if not batched:
        x = x[np.newaxis, :]
    log_c = np.log(np.maximum(x, 1e-8))
    n_steps = log_c.shape[1]
    t_arr = np.arange(n_steps, dtype=np.float64) * dt
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


def coherence_envelope(coh: np.ndarray) -> np.ndarray:
    """Returns the slow decay envelope of L1 coherence.

    L1 coherence is nonnegative, so a Hilbert transform of the raw C(t)
    distorts a decaying carrier. The upper peak envelope is used when the
    window has enough maxima (TFIM-like); otherwise a reflect-padded
    Hilbert envelope.

    Args:
        coh: Coherence values, shape (T,) or (N, T).

    Returns:
        Envelope with the shape of `coh`, bounded below by 1e-8.
    """
    x = np.asarray(coh, dtype=np.float64)
    batched = x.ndim == 2
    if not batched:
        x = x[np.newaxis, :]
    out = np.empty_like(x)
    n_steps = x.shape[1]
    idx = np.arange(n_steps)
    for i, row in enumerate(x):
        peaks, _ = signal.find_peaks(row, distance=2)
        extra = []
        if row[0] >= row[1]:
            extra.append(0)
        if row[-1] >= row[-2]:
            extra.append(n_steps - 1)
        if extra:
            peaks = np.unique(
                np.concatenate([peaks, np.array(extra, dtype=int)]))
        if peaks.size >= 3:
            out[i] = np.interp(idx, peaks.astype(np.float64), row[peaks])
            continue
        padded = np.concatenate([row[::-1], row, row[::-1]])
        out[i] = np.abs(signal.hilbert(padded))[n_steps:2 * n_steps]
    out = np.maximum(out, 1e-8)
    return out[0] if not batched else out


def maybe_envelope_slope(
    coh: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fits the raw coherence, or its envelope when that is straighter.

    Args:
        coh: Coherence values, shape (T,) or (N, T).
        dt: Time step between samples.

    Returns:
        (slope, r2) of the chosen fit, per row.
    """
    slopes, r2 = ols_log_slope(coh, dt)
    env_slopes, env_r2 = ols_log_slope(coherence_envelope(coh), dt)
    use_env = (r2 < ENVELOPE_RAW_R2) & (env_r2 > r2)
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
) -> tuple[float, float, float]:
    """Estimates the remaining decoherence time from one window.

    For a log-linear decay T2 = -1/slope, so the remaining time is
    -1/slope - t_obs. On oscillatory windows (raw OLS R^2 below
    `ENVELOPE_RAW_R2`) the slope comes from the envelope of C(t) when that
    fit is straighter; the identity is the same but on the slow decay mode
    instead of the carrier.

    Args:
        window: Observables, shape (T, F); the last column is coherence_l1.
        t_obs: Absolute time at the end of the window.
        dt: Time step between rows.
        t_max: Upper bound on the estimate.
        adaptive_r2_threshold: If positive, an estimate whose OLS R^2 is
            below it is replaced by 0 so that a learned residual takes
            over. 0.7 works for two-qubit systems.

    Returns:
        (remaining_estimate, slope, ols_r2).
    """
    coh = np.abs(window[:, -1]) + 1e-8
    slope, ols_r2 = maybe_envelope_slope(coh, dt)
    slope = float(slope)
    ols_r2 = float(ols_r2)

    no_decay = slope >= _FLAT_SLOPE
    unreliable = adaptive_r2_threshold > 0.0 and ols_r2 < adaptive_r2_threshold
    if no_decay or unreliable:
        return 0.0, slope, ols_r2

    raw_est = -1.0 / slope - t_obs
    remaining_est = float(np.clip(raw_est, 0.0, t_max))
    return remaining_est, slope, ols_r2


def batch_physics(
    windows: np.ndarray,
    feature_names: list[str],
    dt: float,
    t_obs: np.ndarray,
    t_max: float = 20.0,
    adaptive_r2_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the physics scalars for an array of windows.

    High OLS R^2 means log-linear decay (XXZ-like); low R^2 means
    oscillation (TFIM-like). With `adaptive_r2_threshold` > 0 the estimate
    of windows below the threshold is 0, so that the network takes full
    responsibility on oscillatory two-qubit coherence.

    Args:
        windows: Observables, shape (N, T, F).
        feature_names: Names of the F columns; `coherence_l1` is looked up
            by name and defaults to the last column.
        dt: Time step between rows.
        t_obs: Observation time of each window, shape (N,).
        t_max: Upper bound on the remaining-time estimate.
        adaptive_r2_threshold: See above.

    Returns:
        (slopes, log_coherence_at_obs, remaining_estimate, ols_r2), each
        float32 of shape (N,).
    """
    names = list(feature_names)
    ci = names.index("coherence_l1") if "coherence_l1" in names \
        else windows.shape[-1] - 1

    coh = np.abs(windows[:, :, ci]) + 1e-8
    log_coh_at_obs = np.log(coh[:, -1])
    slopes, ols_r2 = maybe_envelope_slope(coh, dt)

    if adaptive_r2_threshold > 0.0:
        bad_prior = (slopes >= _FLAT_SLOPE) | (ols_r2 < adaptive_r2_threshold)
    else:
        bad_prior = slopes >= _FLAT_SLOPE

    raw_rem = np.where(
        ~bad_prior, -1.0 / np.where(bad_prior, -1.0, slopes) - t_obs, 0.0)
    phys_rem = np.clip(raw_rem, 0.0, t_max)

    return (slopes.astype(np.float32), log_coh_at_obs.astype(np.float32),
            phys_rem.astype(np.float32), ols_r2.astype(np.float32))
