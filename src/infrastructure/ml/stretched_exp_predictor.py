"""Classical baseline: stretched-exponential fit C(t) = C0 * exp(-(t/tau)**beta).

With the relative 1/e threshold, coherence crosses C0/e exactly at t = tau,
so the predicted decoherence time is simply the fitted tau.
No training required (fit happens per-window at predict time).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import curve_fit

from ...domain.ports import IPredictor
from ...domain.value_objects import PredictionResult


def _stretched_exp(t, log_c0, tau, beta):
    return log_c0 - (t / tau) ** beta


class StretchedExpPredictor(IPredictor):
    """curve_fit baseline — no learnable parameters."""

    def __init__(self, t_max: float = 20.0, dt: float = 0.1) -> None:
        self.t_max = t_max
        self.dt = dt

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, dataset, **kwargs):
        """No-op: the fit is performed per window at predict time."""
        return self

    def predict(self, window: np.ndarray, t_obs: float, horizon: float) -> PredictionResult:
        coh = np.abs(window[:, -1]) + 1e-8
        T = len(coh)
        # absolute time axis: window ends at t_obs
        t = t_obs - (T - 1 - np.arange(T)) * self.dt
        t = np.clip(t, 1e-6, None)
        log_coh = np.log(coh)
        try:
            popt, _ = curve_fit(
                _stretched_exp, t, log_coh,
                p0=(float(log_coh[0]), max(t_obs, 1.0), 1.0),
                bounds=([-20.0, 1e-3, 0.2], [5.0, 10.0 * self.t_max, 3.0]),
                maxfev=2000,
            )
            tau = float(popt[1])
        except Exception:
            tau = self.t_max  # fit failed: pessimistic fallback
        t_decoh_pred = float(np.clip(tau, t_obs, self.t_max))
        remaining = t_decoh_pred - t_obs
        risk = 1.0 if remaining <= horizon else 0.0
        return PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=risk,
            horizon=horizon,
        )

    def state_bytes(self) -> bytes:
        return json.dumps({"t_max": self.t_max, "dt": self.dt}).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "StretchedExpPredictor":
        d = json.loads(data.decode())
        return cls(t_max=d["t_max"], dt=d["dt"])
