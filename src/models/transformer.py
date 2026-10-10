"""Pure-Transformer decoherence predictor (ablation baseline, E1).

The architecture follows arXiv 2505.06928 but is applied directly to the
T2 prediction task, not to the gamma-reconstruction inverse problem. There
is no physics prior: the Transformer must learn the exponential decay
structure from data.

Architecture::

    input (N, T, F)
      -> linear projection to d_model
      -> sinusoidal positional encoding (fixed)
      -> TransformerEncoder (n_layers, n_heads, dim_ff, dropout)
      -> mean pooling over the sequence -> (N, d_model)
      -> regression head: Linear -> GELU -> Dropout -> Linear -> scalar
      -> risk head:       Linear -> GELU -> Dropout -> Linear -> logit

At prediction time the risk logit is unused: the risk is
`remaining_to_risk(remaining, horizon)`.

Compared to `lstm.LSTMPredictor` there is no OLS estimate and no residual
correction (the raw output is the prediction), long-range patterns in the
window are easier to capture, and the weaker inductive bias needs more data
or a larger window.
"""
from __future__ import annotations

import dataclasses
import io
import math

import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data

from src import reproducibility
from src.data import dataset as dataset_lib
from src.domain import ports
from src.domain import value_objects
from src.models import physics_prior
from src.training import loop
from src.training import losses
from src.training import standardize

# Early-stopping patience; the Transformer has no configurable value.
_PATIENCE = 25


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """Precomputes the encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Longest supported sequence.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the encoding to embeddings of shape (N, T, d_model)."""
        return x + self.pe[:, :x.size(1)]


class TransformerNet(nn.Module):
    """Encoder-only Transformer with a regression and a risk head."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Builds the network.

        Args:
            n_features: Input channels per time step.
            d_model: Embedding dimension.
            n_heads: Attention heads.
            n_layers: Encoder layers.
            dim_ff: Feed-forward hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the network.

        Args:
            x: Standardised windows, shape (N, T, n_features).

        Returns:
            (regression, risk_logit), each of shape (N,).
        """
        h = self.pos_enc(self.input_proj(x))   # (N, T, d_model)
        h = self.encoder(h)                    # (N, T, d_model)
        pooled = h.mean(dim=1)                 # (N, d_model)
        reg = self.regression_head(pooled).squeeze(-1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        return reg, risk_logit


@dataclasses.dataclass
class TransformerPredictor(ports.IPredictor):
    """Pure Transformer predictor without a physics prior.

    Attributes:
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        dim_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        t_max: Hard upper bound on the predicted T2.
        net: The trained network, None before training.
        dt: Time step between window rows (stored in checkpoints only).
        feature_scaler: Per-channel standardiser of the windows.
        target_scaler: Standardiser of the remaining time.
        device: Torch device the network runs on.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    t_max: float = 10.0

    net: TransformerNet | None = dataclasses.field(
        default=None, init=False, repr=False)
    dt: float = dataclasses.field(default=0.1, init=False, repr=False)
    feature_scaler: standardize.Standardizer | None = dataclasses.field(
        default=None, init=False, repr=False)
    target_scaler: standardize.Standardizer = dataclasses.field(
        default_factory=standardize.Standardizer, init=False, repr=False)
    device: str = dataclasses.field(default="cpu", init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_trained(self) -> bool:
        return self.net is not None

    def predict(
        self, window: np.ndarray, t_obs: float, horizon: float
    ) -> value_objects.PredictionResult:
        """Predicts the decoherence time of one window.

        Args:
            window: Array of shape (window_length, n_features).
            t_obs: Absolute time at the end of the window.
            horizon: Time window for the risk score.

        Returns:
            The predicted decoherence time and risk score.

        Raises:
            RuntimeError: If the predictor is not trained.
        """
        if self.net is None:
            raise RuntimeError("TransformerPredictor is not trained.")
        self.net.eval()
        with torch.no_grad():
            x = self._standardise_window(window)
            reg_norm, _risk_logit = self.net(x)
            remaining = (float(reg_norm.item()) * self.target_scaler.std
                         + self.target_scaler.mean)
            remaining = max(0.0, remaining)
            t_decoh_pred = float(np.clip(t_obs + remaining, 0.0, self.t_max))
            risk = physics_prior.remaining_to_risk(remaining, horizon)
        return value_objects.PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=risk,
            horizon=horizon,
        )

    def train(self, dataset: object, **kwargs) -> loop.TrainResult:
        """Fits the network to the training split of a dataset.

        Args:
            dataset: A `Dataset`; its train split is fitted and its val
                split drives early stopping.
            **kwargs: See `loop.FitOptions` (n_epochs, batch_size, lr,
                verbose, regression_loss, seed).

        Returns:
            The predictor and the loss histories.

        Raises:
            TypeError: If `dataset` is not a `Dataset`.
        """
        if not isinstance(dataset, dataset_lib.Dataset):
            raise TypeError("TransformerPredictor.train needs a Dataset.")
        options = loop.FitOptions.from_kwargs(kwargs)
        if options.seed is not None:
            # See LSTMPredictor.train: torch's global RNG drives weight
            # init, dropout and batch shuffling.
            reproducibility.seed_torch(options.seed)
        train_hist, val_hist = self._fit(dataset, options)
        return loop.TrainResult(
            predictor=self,
            train_loss_history=train_hist,
            val_loss_history=val_hist,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_bytes(self) -> bytes:
        """Serialises the network, scalers and configuration (torch.save).

        Returns:
            The checkpoint bytes read back by `from_bytes`.

        Raises:
            RuntimeError: If the predictor is not trained.
        """
        if self.net is None:
            raise RuntimeError("Cannot serialise an untrained predictor.")
        buf = io.BytesIO()
        torch.save({
            "state_dict": self.net.state_dict(),
            "feature_mean": self.feature_scaler.mean,
            "feature_std": self.feature_scaler.std,
            "target_mean": self.target_scaler.mean,
            "target_std": self.target_scaler.std,
            "dt": self.dt,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dim_ff": self.dim_ff,
            "dropout": self.dropout,
            "n_features": self.net.input_proj.in_features,
        }, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> TransformerPredictor:
        """Restores a predictor from `state_bytes`.

        Args:
            data: Checkpoint bytes.

        Returns:
            The trained predictor, with its network in eval mode.
        """
        ckpt = torch.load(io.BytesIO(data), map_location="cpu",
                          weights_only=False)
        p = cls(
            d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"], dim_ff=ckpt["dim_ff"],
            dropout=ckpt["dropout"],
        )
        net = TransformerNet(
            n_features=ckpt["n_features"], d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
            dim_ff=ckpt["dim_ff"], dropout=ckpt["dropout"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        p.net = net
        p.feature_scaler = standardize.Standardizer(
            ckpt["feature_mean"], ckpt["feature_std"])
        p.target_scaler = standardize.Standardizer(
            ckpt["target_mean"], ckpt["target_std"])
        p.dt = ckpt.get("dt", 0.1)
        return p

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _fit(
        self, ds: dataset_lib.Dataset, options: loop.FitOptions
    ) -> tuple[list[float], list[float]]:
        train = ds.train_idx
        val = ds.val_idx
        self.feature_scaler = standardize.Standardizer.fit(
            ds.sequences[train], axis=(0, 1))
        self.target_scaler = standardize.Standardizer.fit(
            ds.remaining_time[train])

        self.net = TransformerNet(
            ds.sequences.shape[-1], self.d_model, self.n_heads,
            self.n_layers, self.dim_ff, self.dropout,
        ).to(self.device)

        reg_loss = losses.get_loss(options.regression_loss).as_torch()
        bce = nn.BCEWithLogitsLoss()

        def step(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
            xb, yb, rb = (t.to(self.device) for t in batch)
            reg_pred, risk_pred = self.net(xb)
            return reg_loss(reg_pred, yb) + bce(risk_pred, rb)

        return loop.fit(
            self.net,
            self._loader(ds, train, options.batch_size, True),
            self._loader(ds, val, options.batch_size, False),
            step=step,
            n_epochs=options.n_epochs,
            lr=options.lr,
            patience=_PATIENCE,
            verbose=options.verbose,
            label="Transformer",
        )

    def _loader(
        self,
        ds: dataset_lib.Dataset,
        idx: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> torch_data.DataLoader:
        arrays = (
            self.feature_scaler.apply(ds.sequences[idx]),
            self.target_scaler.apply(ds.remaining_time[idx]),
            ds.risk_labels[idx],
        )
        tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
        return torch_data.DataLoader(
            torch_data.TensorDataset(*tensors),
            batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _standardise_window(self, window: np.ndarray) -> torch.Tensor:
        normed = self.feature_scaler.apply(window)
        return torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(
            self.device)
