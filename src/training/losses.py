"""Loss functions and the loss registry.

Each loss wraps a PyTorch criterion so it can be used by the training loop
(`as_torch()`, an `nn.Module`) and by evaluation code (called on NumPy
arrays). `LOSSES` maps a string alias to a loss; `get_loss` also understands
`"quantile_<q>"` for any q in (0, 1).

Usage::

    loss = losses.get_loss("huber")
    loss(preds_np, targets_np)   # float, NumPy path
    loss.as_torch()              # nn.Module, training path
"""
from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from src.domain import ports


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


@dataclasses.dataclass(frozen=True)
class HuberLossFunction(ports.ILossFunction):
    """Huber (smooth L1) loss, robust to outliers; the default choice.

    Quadratic for |e| <= delta, linear beyond. delta=1.0 is a sensible
    default for normalised residuals.

    Attributes:
        delta: Transition point between the quadratic and linear parts.
    """

    delta: float = 1.0

    @property
    def name(self) -> str:
        return f"huber(delta={self.delta})"

    def __call__(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        criterion = nn.HuberLoss(delta=self.delta, reduction="mean")
        with torch.no_grad():
            loss = criterion(_to_torch(predictions), _to_torch(targets))
        return float(loss.item())

    def as_torch(self) -> nn.Module:
        return nn.HuberLoss(delta=self.delta, reduction="mean")


@dataclasses.dataclass(frozen=True)
class MSELossFunction(ports.ILossFunction):
    """Mean squared error; sensitive to outliers, useful as a check."""

    @property
    def name(self) -> str:
        return "mse"

    def __call__(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        return float(np.mean((predictions - targets) ** 2))

    def as_torch(self) -> nn.Module:
        return nn.MSELoss(reduction="mean")


@dataclasses.dataclass(frozen=True)
class MAELossFunction(ports.ILossFunction):
    """Mean absolute error (L1)."""

    @property
    def name(self) -> str:
        return "mae"

    def __call__(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        return float(np.mean(np.abs(predictions - targets)))

    def as_torch(self) -> nn.Module:
        return nn.L1Loss(reduction="mean")


@dataclasses.dataclass(frozen=True)
class QuantileLossFunction(ports.ILossFunction):
    """Pinball / quantile loss for prediction intervals.

    L_q(y, p) = q * max(y - p, 0) + (1 - q) * max(p - y, 0). Training the
    lower (q < 0.5) and upper (q > 0.5) bounds around the median (q = 0.5)
    gives a prediction interval.

    Attributes:
        quantile: The quantile q in (0, 1).

    Raises:
        ValueError: If `quantile` is not in (0, 1).
    """

    quantile: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be in (0, 1).")

    @property
    def name(self) -> str:
        return f"quantile(q={self.quantile})"

    def __call__(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        e = targets - predictions
        return float(
            np.mean(np.maximum(self.quantile * e, (self.quantile - 1) * e)))

    def as_torch(self) -> nn.Module:
        q = self.quantile

        class _QuantileLoss(nn.Module):

            def forward(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                e = target - pred
                return torch.mean(torch.max(q * e, (q - 1) * e))

        return _QuantileLoss()


class SurvHuberLoss(nn.Module):
    """Huber loss with a one-sided penalty for right-censored targets.

    For a censored target (a lower bound L) the loss is 0 if the prediction
    is at least L and Huber(L - pred) otherwise. This is the SurvLoss of
    PHM 2024 with a Huber envelope.
    """

    def __init__(self, delta: float = 1.0) -> None:
        """Creates the loss.

        Args:
            delta: Huber transition point.
        """
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        censored: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the mean loss.

        Args:
            pred: Predictions.
            target: Targets; lower bounds where `censored` is set.
            censored: 1 where the target is a lower bound, else 0. None
                means no sample is censored.

        Returns:
            Scalar loss.
        """
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


@dataclasses.dataclass(frozen=True)
class SurvHuberLossFunction(ports.ILossFunction):
    """`SurvHuberLoss` as an `ILossFunction`.

    On NumPy arrays (no censoring information) it equals the Huber loss.

    Attributes:
        delta: Huber transition point.
    """

    delta: float = 1.0

    @property
    def name(self) -> str:
        return f"surv_huber(delta={self.delta})"

    def __call__(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        return HuberLossFunction(delta=self.delta)(predictions, targets)

    def as_torch(self) -> nn.Module:
        return SurvHuberLoss(delta=self.delta)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSSES: Mapping[str, ports.ILossFunction] = types.MappingProxyType({
    "huber": HuberLossFunction(),
    "mse": MSELossFunction(),
    "mae": MAELossFunction(),
    "quantile_median": QuantileLossFunction(quantile=0.5),
    "surv_huber": SurvHuberLossFunction(),
})

_QUANTILE_PREFIX = "quantile_"


def get_loss(name: str) -> ports.ILossFunction:
    """Looks up a loss function by alias.

    Args:
        name: A key of `LOSSES`, or "quantile_<q>" with q a float in
            (0, 1), e.g. "quantile_0.9".

    Returns:
        The loss.

    Raises:
        KeyError: If `name` is not a known alias; the message lists the
            valid ones.
    """
    if name in LOSSES:
        return LOSSES[name]
    if name.startswith(_QUANTILE_PREFIX):
        try:
            q = float(name[len(_QUANTILE_PREFIX):])
        except ValueError:
            pass
        else:
            return QuantileLossFunction(quantile=q)
    raise KeyError(
        f"Unknown loss {name!r}. Valid aliases: {sorted(LOSSES)} "
        "and 'quantile_<q>'."
    )
