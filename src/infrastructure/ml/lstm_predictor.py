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


# ---------------------------------------------------------------------------
# Neural network (residual corrector on top of physics estimate)
# ---------------------------------------------------------------------------

class _ResidualNet(nn.Module):
    """BiLSTM that predicts residual corrections to the physics estimate."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        n_physics: int = 7,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # physics scalars: t_obs, slope, log_coh, physics_remaining, ols_r2,
        #                  interaction_code, dissipator_code, [optional J]
        self.n_physics = n_physics
        enc_dim = hidden_size * 2 + n_physics

        self.regression_head = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_norm: torch.Tensor,
        slope_norm: torch.Tensor,
        logc_norm: torch.Tensor,
        phys_norm: torch.Tensor,
        r2_norm: torch.Tensor,
        int_norm: torch.Tensor,
        dis_norm: torch.Tensor,
        j_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        extras = [
            t_norm.unsqueeze(1),
            slope_norm.unsqueeze(1),
            logc_norm.unsqueeze(1),
            phys_norm.unsqueeze(1),
            r2_norm.unsqueeze(1),
            int_norm.unsqueeze(1),
            dis_norm.unsqueeze(1),
        ]
        if j_norm is not None:
            extras.append(j_norm.unsqueeze(1))
        extras = extras[: self.n_physics]
        pooled = torch.cat([pooled, *extras], dim=1)

        residual  = self.regression_head(pooled).squeeze(-1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        return residual, risk_logit


# ---------------------------------------------------------------------------
# Predictor adapter
# ---------------------------------------------------------------------------

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
        from ...application.generate_dataset import Dataset as DS
        ds: DS = ds

        # Sync dt from dataset so physics formulas use the actual simulation step.
        self._dt = float(ds.metadata.get("dt", self._dt))

        def prep(idx):
            X   = ds.sequences[idx]
            rem = ds.remaining_time[idx]
            t   = ds.t_obs[idx]
            r   = ds.risk_labels[idx]
            slopes, logcs, phys, r2s = _batch_physics(
                X, ds.feature_names, self._dt, t, self.t_max,
                adaptive_r2_threshold=self.adaptive_prior_r2_threshold,
            )
            if not self.use_physics_prior:
                phys    = np.zeros_like(phys)
                slopes  = np.zeros_like(slopes)
            residuals = (rem - phys).astype(np.float32)
            X_aug = _add_log(X, ds.feature_names)
            # Context codes: use dataset arrays if available, else zeros.
            int_codes = (ds.interaction_type_codes[idx]
                         if len(ds.interaction_type_codes) == len(ds.sequences)
                         else np.zeros(len(idx), dtype=np.float32))
            dis_codes = (ds.dissipator_type_codes[idx]
                         if len(ds.dissipator_type_codes) == len(ds.sequences)
                         else np.zeros(len(idx), dtype=np.float32))
            j_vals = (ds.J_values[idx]
                      if len(getattr(ds, "J_values", [])) == len(ds.sequences)
                      else np.zeros(len(idx), dtype=np.float32))
            if len(getattr(ds, "censored", [])) == len(ds.sequences):
                cens = ds.censored[idx].astype(np.float32)
            else:
                cens = np.zeros(len(idx), dtype=np.float32)
            return X_aug, rem, t, slopes, logcs, phys, r2s, int_codes, dis_codes, j_vals, residuals, r, cens

        X_tr, rem_tr, t_tr, sl_tr, lc_tr, ph_tr, r2_tr, ic_tr, dc_tr, j_tr, res_tr, r_tr, c_tr = prep(ds.train_idx)
        X_va, rem_va, t_va, sl_va, lc_va, ph_va, r2_va, ic_va, dc_va, j_va, res_va, r_va, c_va = prep(ds.val_idx)

        # normalisation
        self._feature_mean = X_tr.mean(axis=(0,1))
        self._feature_std  = X_tr.std(axis=(0,1)) + 1e-8
        self._t_obs_mean    = float(t_tr.mean())
        self._t_obs_std     = float(t_tr.std()) + 1e-8
        self._slope_mean    = float(sl_tr.mean())
        self._slope_std     = float(sl_tr.std()) + 1e-8
        self._logc_mean     = float(lc_tr.mean())
        self._logc_std      = float(lc_tr.std()) + 1e-8
        self._phys_mean     = float(ph_tr.mean())
        self._phys_std      = float(ph_tr.std()) + 1e-8
        self._r2_mean       = float(r2_tr.mean())
        self._r2_std        = float(r2_tr.std()) + 1e-8
        self._ic_mean       = float(ic_tr.mean())
        self._ic_std        = float(ic_tr.std()) + 1e-8
        self._dc_mean       = float(dc_tr.mean())
        self._dc_std        = float(dc_tr.std()) + 1e-8
        self._j_mean        = float(j_tr.mean())
        self._j_std         = float(j_tr.std()) + 1e-8
        unc_tr = c_tr < 0.5
        res_for_stats = res_tr[unc_tr] if unc_tr.any() else res_tr
        self._residual_mean = float(res_for_stats.mean())
        self._residual_std  = float(res_for_stats.std()) + 1e-8

        n_feat = X_tr.shape[-1]
        n_physics = 8 if self.use_J_scalar else 7
        self._net = _ResidualNet(
            n_feat, self.hidden_size, self.num_layers, self.dropout, n_physics=n_physics,
        ).to(self._device)

        opt   = torch.optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
        from .loss_functions import SurvHuberLoss, get_loss
        use_surv = bool(c_tr.any())
        _reg_name = "huber" if regression_loss == "surv_huber" else regression_loss
        reg_loss_fn = get_loss(_reg_name).as_torch()
        surv_fn = SurvHuberLoss(delta=1.0)
        horizon = float(ds.metadata.get("horizon", 1.0))

        def _reg_loss(res_pred, yn, pb, remb, cb):
            if not use_surv:
                return reg_loss_fn(res_pred, yn)
            pred_rem = pb + res_pred * self._residual_std + self._residual_mean
            unc = cb < 0.5
            acc = res_pred.new_zeros(())
            if unc.any():
                acc = acc + reg_loss_fn(res_pred[unc], yn[unc]) * unc.sum()
            if (~unc).any():
                ones = torch.ones_like(cb[~unc])
                acc = acc + surv_fn(pred_rem[~unc], remb[~unc], ones) * (~unc).sum()
            return acc / cb.numel()

        # Weighted BCE: up-weight the positive (imminent-decoherence) class.
        # risk_pos_weight == 0 → auto-compute from label ratio; clamp to [1, 10].
        if self.risk_pos_weight == 0.0:
            n_pos = float(r_tr.sum()) + 1e-3
            n_neg = float(len(r_tr) - n_pos) + 1e-3
            _pw = float(np.clip(n_neg / n_pos, 1.0, 10.0))
        else:
            _pw = float(self.risk_pos_weight)
        _pw_tensor = torch.tensor(_pw, dtype=torch.float32)

        def _bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return nn.functional.binary_cross_entropy_with_logits(
                pred, target, pos_weight=_pw_tensor.to(pred.device),
            )

        tr_ld = _loader(X_tr, res_tr, t_tr, sl_tr, lc_tr, ph_tr, r2_tr, ic_tr, dc_tr, j_tr, r_tr, rem_tr, c_tr, batch_size, True)
        va_ld = _loader(X_va, res_va, t_va, sl_va, lc_va, ph_va, r2_va, ic_va, dc_va, j_va, r_va, rem_va, c_va, batch_size, False)

        train_hist, val_hist = [], []
        best_val, best_st, pat = float("inf"), None, 0
        patience = self.early_stopping_patience

        for ep in range(1, n_epochs + 1):
            self._net.train()
            tl = 0.0
            for xb, yb, tb, sb, lb, pb, r2b, ib, db, jb, rb, remb, cb in tr_ld:
                xb = xb.to(self._device)
                rb = rb.to(self._device)
                remb = remb.to(self._device)
                cb = cb.to(self._device)
                pb = pb.to(self._device)
                if self.augment_sigma > 0.0:
                    xb = xb + torch.randn_like(xb) * self.augment_sigma
                yn  = ((yb  - self._residual_mean) / self._residual_std).to(self._device)
                tn  = ((tb  - self._t_obs_mean)    / self._t_obs_std).to(self._device)
                sn  = ((sb  - self._slope_mean)    / self._slope_std).to(self._device)
                ln  = ((lb  - self._logc_mean)     / self._logc_std).to(self._device)
                pn  = ((pb  - self._phys_mean)     / self._phys_std)
                r2n = ((r2b - self._r2_mean)       / self._r2_std).to(self._device)
                itn = ((ib  - self._ic_mean)       / self._ic_std).to(self._device)
                dtn = ((db  - self._dc_mean)       / self._dc_std).to(self._device)
                jn = (
                    ((jb - self._j_mean) / self._j_std).to(self._device)
                    if self.use_J_scalar else None
                )
                res_pred, risk_pred = self._net(xb, tn, sn, ln, pn, r2n, itn, dtn, jn)
                reg = _reg_loss(res_pred, yn, pb, remb, cb)
                risk_valid = ~((cb > 0.5) & (remb <= horizon))
                if risk_valid.any():
                    bce = _bce(risk_pred[risk_valid], rb[risk_valid])
                else:
                    bce = risk_pred.new_zeros(())
                loss = self.alpha * reg + self.beta * bce
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                tl += loss.item()
            avg_tr = tl / max(len(tr_ld), 1)

            self._net.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb, tb, sb, lb, pb, r2b, ib, db, jb, rb, remb, cb in va_ld:
                    xb = xb.to(self._device)
                    rb = rb.to(self._device)
                    remb = remb.to(self._device)
                    cb = cb.to(self._device)
                    pb = pb.to(self._device)
                    yn  = ((yb  - self._residual_mean) / self._residual_std).to(self._device)
                    tn  = ((tb  - self._t_obs_mean)    / self._t_obs_std).to(self._device)
                    sn  = ((sb  - self._slope_mean)    / self._slope_std).to(self._device)
                    ln  = ((lb  - self._logc_mean)     / self._logc_std).to(self._device)
                    pn  = ((pb  - self._phys_mean)     / self._phys_std)
                    r2n = ((r2b - self._r2_mean)       / self._r2_std).to(self._device)
                    itn = ((ib  - self._ic_mean)       / self._ic_std).to(self._device)
                    dtn = ((db  - self._dc_mean)       / self._dc_std).to(self._device)
                    jn = (
                        ((jb - self._j_mean) / self._j_std).to(self._device)
                        if self.use_J_scalar else None
                    )
                    res_pred, risk_pred = self._net(xb, tn, sn, ln, pn, r2n, itn, dtn, jn)
                    reg = _reg_loss(res_pred, yn, pb, remb, cb)
                    risk_valid = ~((cb > 0.5) & (remb <= horizon))
                    if risk_valid.any():
                        bce = _bce(risk_pred[risk_valid], rb[risk_valid])
                    else:
                        bce = risk_pred.new_zeros(())
                    vl += (self.alpha * reg + self.beta * bce).item()
            avg_va = vl / max(len(va_ld), 1)
            sched.step(avg_va)
            train_hist.append(avg_tr)
            val_hist.append(avg_va)

            if avg_va < best_val - 1e-4:
                best_val = avg_va
                best_st  = copy.deepcopy(self._net.state_dict())
                pat = 0
            else:
                pat += 1

            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"  Epoch {ep:3d}/{n_epochs}  train={avg_tr:.4f}  val={avg_va:.4f}"
                      f"  (best={best_val:.4f}  pat={pat}/{patience})")
            if pat >= patience:
                if verbose:
                    print(f"  Early stop at epoch {ep}.")
                break

        if best_st is not None:
            self._net.load_state_dict(best_st)
        self._net.eval()
        return train_hist, val_hist

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
