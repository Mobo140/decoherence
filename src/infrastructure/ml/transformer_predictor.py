"""Pure-Transformer decoherence predictor — ablation baseline (E1).

Architecture mirrors the spirit of arXiv 2505.06928 but applied directly
to the T₂ prediction task (not the γ-reconstruction inverse problem).
There is NO physics prior here — the Transformer must learn everything from
data, including the exponential decay structure.

Architecture::
    input  (N, T, F)
    → linear projection to d_model
    → positional encoding (sinusoidal, fixed)
    → TransformerEncoder (n_layers, n_heads, dim_ff, dropout)
    → CLS-token pooling  (mean over sequence)
    → (N, d_model)
    → regression head:  Linear → GELU → Linear → scalar
    → risk head:        Linear → GELU → Linear → logit (train auxiliary)

  At predict() the risk logit is unused: risk = remaining_to_risk(remaining, horizon).

Compared to LSTMPredictor:
  - No OLS physics estimate embedded
  - No residual correction; raw output is the prediction
  - Better at capturing long-range patterns in the window
  - Weaker inductive bias → needs more data / larger window
"""
from __future__ import annotations

import copy
import io
import math
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
# Positional encoding
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class _TransformerNet(nn.Module):
    """Encoder-only Transformer for decoherence time prediction."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, T, F)
        h = self.pos_enc(self.input_proj(x))   # (N, T, d_model)
        h = self.encoder(h)                    # (N, T, d_model)
        pooled = h.mean(dim=1)                 # (N, d_model)
        reg = self.regression_head(pooled).squeeze(-1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        return reg, risk_logit


# ---------------------------------------------------------------------------
# IPredictor adapter
# ---------------------------------------------------------------------------

@dataclass
class TransformerPredictor(IPredictor):
    """Pure Transformer predictor — no physics prior (ablation baseline).

    Hyper-parameters:
        d_model:   Transformer embedding dimension.
        n_heads:   Number of attention heads.
        n_layers:  Number of encoder layers.
        dim_ff:    Feed-forward hidden dimension.
        dropout:   Dropout rate.
        t_max:     Hard upper bound on predicted T₂.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    t_max: float = 10.0

    _net: Optional[_TransformerNet] = field(default=None, init=False, repr=False)
    _feature_mean: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _feature_std: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _target_mean: float = field(default=0.0, init=False, repr=False)
    _target_std: float = field(default=1.0, init=False, repr=False)
    _dt: float = field(default=0.1, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    def predict(self, window: np.ndarray, t_obs: float, horizon: float) -> PredictionResult:
        if self._net is None:
            raise RuntimeError("TransformerPredictor is not trained.")
        self._net.eval()
        with torch.no_grad():
            x = self._norm_window(window)
            reg_norm, _risk_logit = self._net(x)
            remaining = float(reg_norm.item()) * self._target_std + self._target_mean
            remaining = max(0.0, remaining)
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
        n_epochs   = kwargs.get("n_epochs", 50)
        batch_size = kwargs.get("batch_size", 32)
        lr         = kwargs.get("lr", 1e-3)
        verbose    = kwargs.get("verbose", True)
        loss_name  = kwargs.get("regression_loss", "huber")
        train_hist, val_hist = self._fit(ds, n_epochs, batch_size, lr, verbose, loss_name)
        return TrainModelResult(predictor=self, train_loss_history=train_hist, val_loss_history=val_hist)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_bytes(self) -> bytes:
        if self._net is None:
            raise RuntimeError("Cannot serialise an untrained predictor.")
        buf = io.BytesIO()
        torch.save({
            "state_dict":    self._net.state_dict(),
            "feature_mean":  self._feature_mean,
            "feature_std":   self._feature_std,
            "target_mean":   self._target_mean,
            "target_std":    self._target_std,
            "dt":            self._dt,
            "d_model":       self.d_model,
            "n_heads":       self.n_heads,
            "n_layers":      self.n_layers,
            "dim_ff":        self.dim_ff,
            "dropout":       self.dropout,
            "n_features":    self._net.input_proj.in_features,
        }, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "TransformerPredictor":
        buf  = io.BytesIO(data)
        ckpt = torch.load(buf, map_location="cpu", weights_only=False)
        p = cls(
            d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"], dim_ff=ckpt["dim_ff"],
            dropout=ckpt["dropout"],
        )
        net = _TransformerNet(
            n_features=ckpt["n_features"], d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
            dim_ff=ckpt["dim_ff"], dropout=ckpt["dropout"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        p._net          = net
        p._feature_mean = ckpt["feature_mean"]
        p._feature_std  = ckpt["feature_std"]
        p._target_mean  = ckpt["target_mean"]
        p._target_std   = ckpt["target_std"]
        p._dt           = ckpt.get("dt", 0.1)
        return p

    # ------------------------------------------------------------------
    # Private training
    # ------------------------------------------------------------------

    def _fit(self, ds, n_epochs, batch_size, lr, verbose, loss_name: str):
        from .loss_functions import get_loss

        X_tr = ds.sequences[ds.train_idx]
        X_va = ds.sequences[ds.val_idx]
        rem_tr = ds.remaining_time[ds.train_idx]
        rem_va = ds.remaining_time[ds.val_idx]
        r_tr = ds.risk_labels[ds.train_idx]
        r_va = ds.risk_labels[ds.val_idx]

        self._feature_mean = X_tr.mean(axis=(0, 1))
        self._feature_std  = X_tr.std(axis=(0, 1)) + 1e-8
        self._target_mean  = float(rem_tr.mean())
        self._target_std   = float(rem_tr.std()) + 1e-8

        n_feat = X_tr.shape[-1]
        self._net = _TransformerNet(
            n_feat, self.d_model, self.n_heads, self.n_layers,
            self.dim_ff, self.dropout,
        ).to(self._device)

        opt   = torch.optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        reg_loss_fn = get_loss(loss_name).as_torch()
        bce = nn.BCEWithLogitsLoss()

        def make_loader(X, rem, r, shuffle):
            X_n   = (X - self._feature_mean) / self._feature_std
            rem_n = (rem - self._target_mean) / self._target_std
            td = TensorDataset(
                torch.tensor(X_n,   dtype=torch.float32),
                torch.tensor(rem_n, dtype=torch.float32),
                torch.tensor(r,     dtype=torch.float32),
            )
            return DataLoader(td, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        tr_ld = make_loader(X_tr, rem_tr, r_tr, True)
        va_ld = make_loader(X_va, rem_va, r_va, False)

        train_hist, val_hist = [], []
        best_val, best_st, pat = float("inf"), None, 0
        patience = 25

        for ep in range(1, n_epochs + 1):
            self._net.train()
            tl = 0.0
            for xb, yb, rb in tr_ld:
                xb, yb, rb = xb.to(self._device), yb.to(self._device), rb.to(self._device)
                reg_pred, risk_pred = self._net(xb)
                loss = reg_loss_fn(reg_pred, yb) + bce(risk_pred, rb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                tl += loss.item()
            avg_tr = tl / max(len(tr_ld), 1)

            self._net.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb, rb in va_ld:
                    xb, yb, rb = xb.to(self._device), yb.to(self._device), rb.to(self._device)
                    reg_pred, risk_pred = self._net(xb)
                    vl += (reg_loss_fn(reg_pred, yb) + bce(risk_pred, rb)).item()
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
                print(f"  [Transformer] Epoch {ep:3d}/{n_epochs}  train={avg_tr:.4f}  val={avg_va:.4f}  best={best_val:.4f}  pat={pat}/{patience}")
            if pat >= patience:
                if verbose:
                    print(f"  Early stop at epoch {ep}.")
                break

        if best_st is not None:
            self._net.load_state_dict(best_st)
        self._net.eval()
        return train_hist, val_hist

    def _norm_window(self, window: np.ndarray) -> torch.Tensor:
        normed = (window - self._feature_mean) / self._feature_std
        return torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(self._device)
