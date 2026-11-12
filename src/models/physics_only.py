"""Physics-only predictor: the OLS-slope formula wrapped as an IPredictor.

This is the physics-only ablation baseline (E1 in the paper). It has no
learnable parameters: it computes the decoherence time analytically from the
slope of log(coherence_l1) over the observation window.

For amplitude damping with constant gamma:
    C(t) = C0 * exp(-gamma * t / 2)  ->  log C is linear in t
    slope = d(log C) / dt = -gamma / 2
    T2 = 2 / gamma = -1 / slope
    remaining = T2 - t_obs = -1 / slope - t_obs

For oscillatory two-qubit C(t) the slope is taken from the peak / Hilbert
envelope when that fit is straighter. Other mismatches (time-dependent
gamma) are left to the residual network of `lstm`.

The risk score is the logistic of `physics_prior.remaining_to_risk` with
sharpness `risk_k`.
"""
from __future__ import annotations

import dataclasses
import struct

import numpy as np

from src.domain import ports
from src.domain import value_objects
from src.models import physics_prior


@dataclasses.dataclass
class PhysicsOnlyPredictor(ports.IPredictor):
    """Decoherence predictor using only the OLS log-coherence slope.

    Attributes:
        t_max: Hard upper bound on the predicted T2.
        dt: Time step between consecutive rows in the window.
        risk_k: Sharpness of the logistic risk score.
    """

    t_max: float = 100.0
    dt: float = 0.1
    risk_k: float = 5.0

    @property
    def is_trained(self) -> bool:
        """Always True: the baseline needs no training."""
        return True

    def predict(
        self, window: np.ndarray, t_obs: float, horizon: float
    ) -> value_objects.PredictionResult:
        """Predicts T2 from the OLS slope of log(coherence_l1).

        Args:
            window: Array of shape (window_length, n_features); the last
                column is coherence_l1.
            t_obs: Absolute time at the end of the window.
            horizon: Time window for the risk score.

        Returns:
            The predicted decoherence time and risk score.
        """
        remaining = self._remaining(window, t_obs)
        t_decoh_pred = float(np.clip(t_obs + remaining, 0.0, self.t_max))
        return value_objects.PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=physics_prior.remaining_to_risk(
                remaining, horizon, k=self.risk_k),
            horizon=horizon,
        )

    def train(self, dataset: object, **kwargs) -> object:
        """Does nothing: the formula has no learnable parameters."""
        return self

    def state_bytes(self) -> bytes:
        """Serialises the constructor arguments (nothing is trained)."""
        return struct.pack("ddd", self.t_max, self.dt, self.risk_k)

    @classmethod
    def from_bytes(cls, data: bytes) -> PhysicsOnlyPredictor:
        """Rebuilds a predictor from `state_bytes`."""
        t_max, dt, risk_k = struct.unpack("ddd", data)
        return cls(t_max=t_max, dt=dt, risk_k=risk_k)

    def _remaining(self, window: np.ndarray, t_obs: float) -> float:
        rem, slope, _ = physics_prior.physics_remaining(
            window, t_obs, dt=self.dt, t_max=self.t_max)
        if slope >= -1e-6:
            return float(self.t_max)
        return rem
