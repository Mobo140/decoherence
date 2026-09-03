"""PhysicsOnlyPredictor — OLS-slope formula wrapped as IPredictor.

This is the physics-only ablation baseline (E1 in the paper).  It does NOT
use any neural network; it computes the decoherence time analytically from
the slope of log(coherence_l1) over the observation window.

For amplitude damping (σ₋) with constant γ:
    C(t) = C₀ · exp(−γ·t/2)  →  log C is linear in t
    slope = d(log C)/dt = −γ/2
    T₂ = 2/γ = −1/slope
    remaining = T₂ − t_obs = −1/slope − t_obs

For oscillatory 2-qubit C(t) the slope is taken from the peak / Hilbert
envelope when that fit is straighter.  Other mismatches (time-dependent γ)
are left to the residual BiLSTM.

The risk score is a sigmoid calibrated heuristically:
    risk = σ(−k · (remaining − horizon))   with k tuned to 5.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...domain.ports import IPredictor
from ...domain.value_objects import PredictionResult


def remaining_to_risk(remaining: float, horizon: float, k: float = 5.0) -> float:
    """Map remaining time to P(remaining ≤ horizon).

    Same logistic used by PhysicsOnlyPredictor and, at inference, by the
    LSTM/Transformer: risk = σ(−k · (remaining − horizon)).  Network risk
    heads stay a training auxiliary; their logits are unused at predict().
    """
    z = np.clip(k * (remaining - horizon), -50.0, 50.0)
    return float(1.0 / (1.0 + np.exp(z)))


@dataclass
class PhysicsOnlyPredictor(IPredictor):
    """Decoherence predictor using only the OLS log-coherence slope.

    Parameters:
        t_max:  Hard upper bound on predicted T₂.
        dt:     Time step between consecutive rows in the window.
        risk_k: Sharpness of the sigmoid risk score.
    """

    t_max: float = 100.0
    dt: float = 0.1
    risk_k: float = 5.0

    # Always "trained" — no parameters to fit.
    _fitted: bool = field(default=True, init=False, repr=False)

    @property
    def is_trained(self) -> bool:
        return True

    def predict(self, window: np.ndarray, t_obs: float, horizon: float) -> PredictionResult:
        """Predict T₂ from the OLS slope of log(coherence_l1).

        Args:
            window:  (window_length, n_features) array.  coherence_l1 is
                     assumed to be the last feature column.
            t_obs:   Absolute time at the end of the window.
            horizon: Time window for risk score computation.

        Returns:
            PredictionResult.
        """
        remaining = self._compute_remaining(window, t_obs)
        t_decoh_pred = float(np.clip(t_obs + remaining, 0.0, self.t_max))
        return PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=remaining_to_risk(remaining, horizon, k=self.risk_k),
            horizon=horizon,
        )

    def train(self, dataset: object, **kwargs) -> object:
        """No-op: physics formula has no learnable parameters."""
        return self

    def state_bytes(self) -> bytes:
        import struct
        return struct.pack("ddd", self.t_max, self.dt, self.risk_k)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PhysicsOnlyPredictor":
        import struct
        t_max, dt, risk_k = struct.unpack("ddd", data)
        return cls(t_max=t_max, dt=dt, risk_k=risk_k)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_remaining(self, window: np.ndarray, t_obs: float) -> float:
        """OLS / Hilbert-envelope slope of log(C) → remaining time."""
        from .lstm_predictor import physics_remaining

        rem, slope, _ = physics_remaining(window, t_obs, dt=self.dt, t_max=self.t_max)
        if slope >= -1e-6:
            return float(self.t_max)
        return rem
