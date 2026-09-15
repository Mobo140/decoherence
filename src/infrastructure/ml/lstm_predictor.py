"""Physics-informed decoherence predictor: implements IPredictor via PyTorch.

Primary estimate (pure physics, works for sigma_minus + constant γ + 1/e threshold)
=====================================================================================
The coherence decays as C(t) = C₀·exp(−γ·t/2), so log C is LINEAR in t:

    d(log C)/dt = −γ/2   →   γ = −2·slope

where slope is the OLS slope of log(coherence_l1) over the observation window.
With a 1/e relative threshold the decoherence time is T₂ = 2/γ, hence:

    remaining = T₂ − t_obs = −1/slope − t_obs

This simple formula achieves R²≈0.999 and MAE≈0.05 on held-out data.

Residual LSTM correction (improves generality to time-dependent γ, σ_z, 2-qubit)
===================================================================================
The physics estimate is a perfect starting point but can deviate for:
  - Time-dependent γ(t)  — the slope is only a local estimate
  - σ_z (dephasing only) — different analytical form
  - 2-qubit systems      — entanglement complicates the decay

Architecture
============
  Window (N, T, F) → log(C), log(purity) appended → BiLSTM (hidden=64, bi) → mean pool (128,)
  Physics scalars concatenated (default 7, or 8 if use_J_scalar)::

      t_obs, slope, log C(t_obs), phys_rem, OLS R²,
      interaction_code, dissipator_code, [J]

  Codes (same 4-class as generate_dataset)::

      interaction: 0 = 1q σ−, 1 = 1q σz, 2 = XXZ, 3 = TFIM
      dissipator:  0 = σ−, 1 = σz

  enc_dim = 128 + n_physics  (135 default; old Paper-1 checkpoints had 4 scalars → 132)
  Heads:
      regression : Linear(enc_dim, 64) → GELU → Linear(64, 1)  → residual δT
      risk       : Linear(enc_dim, 32) → GELU → Linear(32, 1)  → logit

  remaining = max(0, phys_rem + δT)   (phys_rem is 0 if use_physics_prior is False)
  T₂̂ = clip(t_obs + remaining, 0, t_max)
  At train time the risk head learns P(remaining ≤ Δt) at the dataset Δt.
  At predict() the logit is unused: risk = remaining_to_risk(remaining, horizon).

"""
from __future__ import annotations

import copy
import io
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ...domain.ports import IPredictor
from ...domain.value_objects import PredictionResult
from .physics_only_predictor import remaining_to_risk

# Re-exported for backward compatibility: these lived in this module before
# task 19 split it up. workbench_server.py imports physics_remaining from here.
from .physics_prior import (  # noqa: F401
    _coherence_envelope,
    _maybe_envelope_slope,
    _n_physics_from_state,
    _ols_log_slope,
    physics_remaining,
)
from .residual_net import _ResidualNet  # noqa: F401
from .batching import _add_log, _batch_physics, _loader, _ns  # noqa: F401

__all__ = ["LSTMPredictor", "physics_remaining"]


@dataclass
class LSTMPredictor(IPredictor):
    """Physics-informed predictor: physics estimate + LSTM residual correction.

    Parameters:
        use_physics_prior: If True (default), the OLS-slope physics estimate is
            embedded as a scalar input and the network predicts the *residual*
            correction on top of it.  If False, the physics estimate is zeroed
            out — the network must learn the full prediction from raw sequences
            (lstm_only ablation baseline, experiment E1).
    """

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.3
    alpha: float = 1.0
    beta: float = 1.0
    t_max: float = 10.0
    use_physics_prior: bool = True
    # Set to e.g. 0.7 for 2-qubit systems to gate the OLS prior by its own R².
    # 0.0 = disabled (backward-compatible default).
    adaptive_prior_r2_threshold: float = 0.0
    # Context codes at inference — must match generate_dataset 4-class scheme:
    #   0.0 = 1q σ−,  1.0 = 1q σz,  2.0 = 2q XXZ,  3.0 = 2q TFIM
    inference_interaction_code: float = 0.0
    # 0.0 = σ− (amplitude damping), 1.0 = σ_z (dephasing)
    inference_dissipator_code: float = 0.0
    # If True, concat normalized coupling J as an 8th physics scalar.
    # Default False keeps the 7-scalar net so old .pt files still load.
    use_J_scalar: bool = False
    # Per-window J at inference; BacktestUseCase sets this from Dataset.J_values.
    inference_J: float = 0.0
    # Positive-class weight for risk BCE loss (0.0 = auto from label ratio, clamped to [1, 10]).
    # Values > 1 up-weight the imminent-decoherence class, improving AUROC for imbalanced labels.
    risk_pos_weight: float = 1.0
    # If > 0, Gaussian noise N(0, augment_sigma) is added to input features during training only.
    # Acts as implicit regularisation, improving cross-system generalisation (≈ weight-noise dropout).
    augment_sigma: float = 0.0
    # Early-stopping patience in epochs. Increase for large datasets (>2000 samples)
    # where each epoch carries more gradient signal and convergence is slower.
    early_stopping_patience: int = 25

    _net: Optional[_ResidualNet] = field(default=None, init=False, repr=False)
    _feature_mean: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _feature_std: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _t_obs_mean: float = field(default=0.0, init=False, repr=False)
    _t_obs_std: float = field(default=1.0, init=False, repr=False)
    _slope_mean: float = field(default=0.0, init=False, repr=False)
    _slope_std: float = field(default=1.0, init=False, repr=False)
    _logc_mean: float = field(default=0.0, init=False, repr=False)
    _logc_std: float = field(default=1.0, init=False, repr=False)
    _phys_mean: float = field(default=0.0, init=False, repr=False)
    _phys_std: float = field(default=1.0, init=False, repr=False)
    _r2_mean: float = field(default=0.0, init=False, repr=False)
    _r2_std: float = field(default=1.0, init=False, repr=False)
    _ic_mean: float = field(default=0.0, init=False, repr=False)
    _ic_std: float = field(default=1.0, init=False, repr=False)
    _dc_mean: float = field(default=0.0, init=False, repr=False)
    _dc_std: float = field(default=1.0, init=False, repr=False)
    _j_mean: float = field(default=0.0, init=False, repr=False)
    _j_std: float = field(default=1.0, init=False, repr=False)
    _residual_mean: float = field(default=0.0, init=False, repr=False)
    _residual_std: float = field(default=1.0, init=False, repr=False)
    _dt: float = field(default=0.1, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    def predict(self, window: np.ndarray, t_obs: float, horizon: float) -> PredictionResult:
        if self._net is None:
            raise RuntimeError("LSTMPredictor is not trained.")

        phys_rem, slope, ols_r2 = physics_remaining(
            window, t_obs, self._dt,
            t_max=self.t_max,
            adaptive_r2_threshold=self.adaptive_prior_r2_threshold,
        )
        if not self.use_physics_prior:
            phys_rem = 0.0
            slope = 0.0
        log_coh = float(np.log(np.abs(window[-1, -1]) + 1e-8))

        self._net.eval()
        with torch.no_grad():
            x = self._norm_window(window)
            t_n   = _ns(t_obs,    self._t_obs_mean,  self._t_obs_std,   self._device)
            sl_n  = _ns(slope,    self._slope_mean,  self._slope_std,   self._device)
            lc_n  = _ns(log_coh,  self._logc_mean,   self._logc_std,    self._device)
            ph_n  = _ns(phys_rem, self._phys_mean,   self._phys_std,    self._device)
            r2_n  = _ns(ols_r2,   self._r2_mean,     self._r2_std,      self._device)
            it_n  = _ns(self.inference_interaction_code, self._ic_mean, self._ic_std, self._device)
            dt_n  = _ns(self.inference_dissipator_code,  self._dc_mean, self._dc_std, self._device)
            j_n = (
                _ns(self.inference_J, self._j_mean, self._j_std, self._device)
                if self.use_J_scalar else None
            )

            residual_norm, _risk_logit = self._net(
                x, t_n, sl_n, lc_n, ph_n, r2_n, it_n, dt_n, j_n,
            )

            residual = float(residual_norm.item()) * self._residual_std + self._residual_mean
            remaining = max(0.0, phys_rem + residual)
            t_decoh_pred = float(np.clip(t_obs + remaining, 0.0, self.t_max))
            risk = remaining_to_risk(remaining, horizon)

        return PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=risk,
            horizon=horizon,
        )

    def train(self, dataset: object, **kwargs) -> object:
        from ...application.train_model import TrainModelResult
        from ...application.generate_dataset import Dataset as DS
        assert isinstance(dataset, DS)
        ds: DS = dataset
        n_epochs        = kwargs.get("n_epochs", 50)
        batch_size      = kwargs.get("batch_size", 32)
        lr              = kwargs.get("lr", 1e-3)
        verbose         = kwargs.get("verbose", True)
        regression_loss = kwargs.get("regression_loss", "huber")
        train_hist, val_hist = self._fit(ds, n_epochs, batch_size, lr, verbose, regression_loss)
        return TrainModelResult(predictor=self, train_loss_history=train_hist, val_loss_history=val_hist)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_bytes(self) -> bytes:
        if self._net is None:
            raise RuntimeError("Cannot serialise an untrained predictor.")
        buf = io.BytesIO()
        torch.save({
            "state_dict":        self._net.state_dict(),
            "feature_mean":      self._feature_mean,
            "feature_std":       self._feature_std,
            "t_obs_mean":        self._t_obs_mean,    "t_obs_std":    self._t_obs_std,
            "slope_mean":        self._slope_mean,    "slope_std":    self._slope_std,
            "logc_mean":         self._logc_mean,     "logc_std":     self._logc_std,
            "phys_mean":         self._phys_mean,     "phys_std":     self._phys_std,
            "r2_mean":           self._r2_mean,       "r2_std":       self._r2_std,
            "ic_mean":           self._ic_mean,       "ic_std":       self._ic_std,
            "dc_mean":           self._dc_mean,       "dc_std":       self._dc_std,
            "residual_mean":     self._residual_mean, "residual_std": self._residual_std,
            "dt":                           self._dt,
            "hidden_size":                  self.hidden_size,
            "num_layers":                   self.num_layers,
            "dropout":                      self.dropout,
            "n_features":                   self._net.lstm.input_size,
            "use_physics_prior":            self.use_physics_prior,
            "adaptive_prior_r2_threshold":  self.adaptive_prior_r2_threshold,
            "risk_pos_weight":              self.risk_pos_weight,
            "augment_sigma":                self.augment_sigma,
            "inference_interaction_code":   self.inference_interaction_code,
            "inference_dissipator_code":    self.inference_dissipator_code,
            "use_J_scalar":                 self.use_J_scalar,
            "inference_J":                  self.inference_J,
            "j_mean":                       self._j_mean,
            "j_std":                        self._j_std,
        }, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LSTMPredictor":
        buf  = io.BytesIO(data)
        ckpt = torch.load(buf, map_location="cpu", weights_only=False)
        p = cls(
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            use_physics_prior=ckpt.get("use_physics_prior", True),
            adaptive_prior_r2_threshold=ckpt.get("adaptive_prior_r2_threshold", 0.0),
            risk_pos_weight=ckpt.get("risk_pos_weight", 1.0),
            augment_sigma=ckpt.get("augment_sigma", 0.0),
            inference_interaction_code=ckpt.get("inference_interaction_code", 0.0),
            inference_dissipator_code=ckpt.get("inference_dissipator_code", 0.0),
            use_J_scalar=ckpt.get("use_J_scalar", False),
            inference_J=ckpt.get("inference_J", 0.0),
        )
        n_physics = _n_physics_from_state(ckpt["state_dict"], ckpt["hidden_size"])
        if n_physics is None:
            n_physics = 8 if p.use_J_scalar else 7
        p.use_J_scalar = n_physics >= 8
        net = _ResidualNet(n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
                           num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
                           n_physics=n_physics)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        p._net           = net
        p._feature_mean  = ckpt["feature_mean"]
        p._feature_std   = ckpt["feature_std"]
        p._t_obs_mean    = ckpt["t_obs_mean"]
        p._t_obs_std     = ckpt["t_obs_std"]
        p._slope_mean    = ckpt["slope_mean"]
        p._slope_std     = ckpt["slope_std"]
        p._logc_mean     = ckpt["logc_mean"]
        p._logc_std      = ckpt["logc_std"]
        p._phys_mean     = ckpt["phys_mean"]
        p._phys_std      = ckpt["phys_std"]
        p._r2_mean       = ckpt.get("r2_mean", 0.0)
        p._r2_std        = ckpt.get("r2_std", 1.0)
        p._ic_mean       = ckpt.get("ic_mean", 0.0)
        p._ic_std        = ckpt.get("ic_std", 1.0)
        p._dc_mean       = ckpt.get("dc_mean", 0.0)
        p._dc_std        = ckpt.get("dc_std", 1.0)
        p._j_mean        = ckpt.get("j_mean", 0.0)
        p._j_std         = ckpt.get("j_std", 1.0)
        p._residual_mean = ckpt["residual_mean"]
        p._residual_std  = ckpt["residual_std"]
        p._dt            = ckpt.get("dt", 0.1)
        return p

    # ------------------------------------------------------------------
    # Private training
    # ------------------------------------------------------------------

    def _fit(self, ds, n_epochs, batch_size, lr, verbose, regression_loss: str = "huber"):
        """Delegates to training.fit_predictor (split out in task 19)."""
        from .training import fit_predictor
        return fit_predictor(self, ds, n_epochs, batch_size, lr, verbose, regression_loss)

    def _norm_window(self, window: np.ndarray) -> torch.Tensor:
        # Pad raw features to the dimension seen at training time.
        # coherence_l1 and purity are always last 2 raw columns; insert zeros
        # before them (same convention as generate_dataset padding).
        n_raw_expected = self._net.lstm.input_size - 2  # _add_log appends 2 cols
        if window.shape[-1] < n_raw_expected:
            n_pad = n_raw_expected - window.shape[-1]
            window = np.concatenate(
                [window[:, :-2],
                 np.zeros((window.shape[0], n_pad), dtype=window.dtype),
                 window[:, -2:]],
                axis=-1,
            )
        w = _add_log(window[np.newaxis], [])[0]
        normed = (w - self._feature_mean) / self._feature_std
        return torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(self._device)

