"""Classical baseline: stretched-exponential fit of the coherence.

C(t) = C0 * exp(-(t / tau) ** beta). With the relative 1/e threshold the
coherence crosses C0/e exactly at t = tau, so the predicted decoherence time
is the fitted tau. Nothing is trained: the fit runs per window at predict
time.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import optimize

from src.domain import ports
from src.domain import value_objects


def _log_stretched_exp(t, log_c0, tau, beta):
    return log_c0 - (t / tau) ** beta


class StretchedExpPredictor(ports.IPredictor):
    """curve_fit baseline without learnable parameters.

    Attributes:
        t_max: Upper bound on the predicted T2.
        dt: Time step between consecutive rows in the window.
    """

    def __init__(self, t_max: float = 20.0, dt: float = 0.1) -> None:
        self.t_max = t_max
        self.dt = dt

    @property
    def is_trained(self) -> bool:
        """Always True: the baseline needs no training."""
        return True

    def train(self, dataset: object, **kwargs) -> object:
        """Does nothing: the fit is performed per window at predict time."""
        return self

    def predict(
        self, window: np.ndarray, t_obs: float, horizon: float
    ) -> value_objects.PredictionResult:
        """Fits tau to the window and returns it as the T2 estimate.

        Args:
            window: Array of shape (window_length, n_features); the last
                column is coherence_l1.
            t_obs: Absolute time at the end of the window.
            horizon: Time window for the risk score.

        Returns:
            The predicted decoherence time; the risk score is 1 if the
            remaining time is within the horizon and 0 otherwise.
        """
        coh = np.abs(window[:, -1]) + 1e-8
        n_steps = len(coh)
        # The window ends at t_obs, so its absolute time axis is shifted.
        t = t_obs - (n_steps - 1 - np.arange(n_steps)) * self.dt
        t = np.clip(t, 1e-6, None)
        log_coh = np.log(coh)
        try:
            popt, _ = optimize.curve_fit(
                _log_stretched_exp, t, log_coh,
                p0=(float(log_coh[0]), max(t_obs, 1.0), 1.0),
                bounds=([-20.0, 1e-3, 0.2], [5.0, 10.0 * self.t_max, 3.0]),
                maxfev=2000,
            )
            tau = float(popt[1])
        except Exception:  # curve_fit raises several unrelated types.
            tau = self.t_max  # Pessimistic fallback when the fit fails.
        t_decoh_pred = float(np.clip(tau, t_obs, self.t_max))
        remaining = t_decoh_pred - t_obs
        risk = 1.0 if remaining <= horizon else 0.0
        return value_objects.PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=risk,
            horizon=horizon,
        )

    def state_bytes(self) -> bytes:
        """Serialises the constructor arguments (nothing is trained)."""
        return json.dumps({"t_max": self.t_max, "dt": self.dt}).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> StretchedExpPredictor:
        """Rebuilds a predictor from `state_bytes`."""
        d = json.loads(data.decode())
        return cls(t_max=d["t_max"], dt=d["dt"])
