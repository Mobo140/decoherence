"""Concrete ILossFunction implementations.

Each class wraps a PyTorch criterion so it can be used both by the
training loop (as nn.Module) and by evaluation code (as a plain callable
on numpy arrays).  All are registered under a string ``name`` for config-
driven selection.

Usage::
    loss = HuberLossFunction(delta=1.0)
    loss(preds_np, targets_np)          # → float (numpy path)
    loss.as_torch()                     # → nn.Module  (training path)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from ...domain.ports import ILossFunction


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------

def _to_torch(arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Concrete losses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HuberLossFunction(ILossFunction):
    """Huber (smooth L1) loss — robust to outliers, default choice.

    Quadratic for |e| ≤ delta, linear beyond.  delta=1.0 is a sensible
    default for normalised residuals.
    """

    delta: float = 1.0
    _registry_name: ClassVar[str] = "huber"

    @property
    def name(self) -> str:
        return f"huber(delta={self.delta})"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        criterion = nn.HuberLoss(delta=self.delta, reduction="mean")
        with torch.no_grad():
            return float(criterion(_to_torch(predictions), _to_torch(targets)).item())

    def as_torch(self) -> nn.Module:
        return nn.HuberLoss(delta=self.delta, reduction="mean")


@dataclass(frozen=True)
class MSELossFunction(ILossFunction):
    """Mean squared error — sensitive to outliers, useful as sanity check."""

    _registry_name: ClassVar[str] = "mse"

    @property
    def name(self) -> str:
        return "mse"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean((predictions - targets) ** 2))

    def as_torch(self) -> nn.Module:
        return nn.MSELoss(reduction="mean")


@dataclass(frozen=True)
class MAELossFunction(ILossFunction):
    """Mean absolute error (L1)."""

    _registry_name: ClassVar[str] = "mae"

    @property
    def name(self) -> str:
        return "mae"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(np.abs(predictions - targets)))

    def as_torch(self) -> nn.Module:
        return nn.L1Loss(reduction="mean")


@dataclass(frozen=True)
class QuantileLossFunction(ILossFunction):
    """Pinball / quantile loss for probabilistic prediction intervals.

    L_q(y, ŷ) = q * max(y−ŷ, 0) + (1−q) * max(ŷ−y, 0)

    Useful for training the lower (q<0.5) and upper (q>0.5) bounds of a
    prediction interval around the median (q=0.5).
    """

    quantile: float = 0.5

    def __post_init__(self) -> None:
        if not (0 < self.quantile < 1):
            raise ValueError("quantile must be in (0, 1).")

    @property
    def name(self) -> str:
        return f"quantile(q={self.quantile})"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        e = targets - predictions
        return float(np.mean(np.maximum(self.quantile * e, (self.quantile - 1) * e)))

    def as_torch(self) -> nn.Module:
        q = self.quantile

        class _QuantileLoss(nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                e = target - pred
                return torch.mean(torch.max(q * e, (q - 1) * e))

        return _QuantileLoss()


@dataclass(frozen=True)
class SurvHuberLossFunction(ILossFunction):
    """Huber on uncensored samples; one-sided Huber on right-censored ones.

    For a censored target (lower bound L): loss is 0 if pred ≥ L, otherwise
    Huber(L − pred).  Matches SurvLoss (PHM 2024) with a Huber envelope.
    """

    delta: float = 1.0
    _registry_name: ClassVar[str] = "surv_huber"

    @property
    def name(self) -> str:
        return f"surv_huber(delta={self.delta})"

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return HuberLossFunction(delta=self.delta)(predictions, targets)

    def as_torch(self) -> nn.Module:
        return SurvHuberLoss(delta=self.delta)


class SurvHuberLoss(nn.Module):
    """Torch module: forward(pred, target, censored=None)."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        censored: torch.Tensor | None = None,
    ) -> torch.Tensor:
        err = pred - target
        abs_e = err.abs()
        huber = torch.where(
            abs_e <= self.delta,
            0.5 * err ** 2,
            self.delta * (abs_e - 0.5 * self.delta),
        )
        if censored is None:
            return huber.mean()
        under = torch.relu(target - pred)
        cens_l = torch.where(
            under <= self.delta,
            0.5 * under ** 2,
            self.delta * (under - 0.5 * self.delta),
        )
        return torch.where(censored.bool(), cens_l, huber).mean()


# ---------------------------------------------------------------------------
# Registry for config-driven lookup
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ILossFunction] = {
    "huber": HuberLossFunction(),
    "mse": MSELossFunction(),
    "mae": MAELossFunction(),
    "quantile_median": QuantileLossFunction(quantile=0.5),
    "surv_huber": SurvHuberLossFunction(),
}


def get_loss(name: str) -> ILossFunction:
    """Look up a loss function by name.

    Args:
        name: One of 'huber', 'mse', 'mae', 'quantile_median', or
              'quantile_<q>' where q is a float in (0, 1) e.g. 'quantile_0.9'.

    Returns:
        ILossFunction instance.

    Raises:
        KeyError: if name is not recognised.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name.startswith("quantile_"):
        q = float(name.split("_", 1)[1])
        return QuantileLossFunction(quantile=q)
    raise KeyError(
        f"Unknown loss '{name}'. Available: {list(_REGISTRY)} + 'quantile_<q>'."
    )
