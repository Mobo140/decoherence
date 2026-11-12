"""The training loop shared by the LSTM and the Transformer.

`fit` runs epochs of AdamW with gradient clipping, a ReduceLROnPlateau
schedule on the validation loss, early stopping and restoration of the best
weights. What differs between models is only how a batch becomes a loss, so
that is passed in as a step function; everything else is the same code.
"""
from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from src.domain import ports

# A step maps one batch (a tuple of tensors from the DataLoader) to the
# scalar loss that is optimised (train) or reported (validation).
StepFn = Callable[[tuple[torch.Tensor, ...]], torch.Tensor]

WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 10
# Smallest improvement of the validation loss that counts as progress.
MIN_DELTA = 1e-4


@dataclasses.dataclass
class TrainResult:
    """Summary produced by training a predictor.

    Attributes:
        predictor: The trained predictor.
        train_loss_history: Mean training loss per epoch.
        val_loss_history: Mean validation loss per epoch.
    """

    predictor: ports.IPredictor
    train_loss_history: list[float] = dataclasses.field(default_factory=list)
    val_loss_history: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class FitOptions:
    """Options common to every trainable predictor.

    Attributes:
        n_epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        lr: Initial AdamW learning rate.
        verbose: Whether to print progress.
        regression_loss: Alias in `losses.LOSSES` for the regression term.
        seed: If not None, passed to `torch.manual_seed` before the
            network is built, so that weight initialisation, dropout and
            batch shuffling are reproducible.
    """

    n_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    verbose: bool = True
    regression_loss: str = "huber"
    seed: int | None = None

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> FitOptions:
        """Reads the options out of `IPredictor.train` keyword arguments.

        Args:
            kwargs: Keyword arguments; unknown keys are ignored.

        Returns:
            Options with defaults for the missing keys.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in names})


def _train_epoch(
    net: nn.Module,
    loader: Any,
    step: StepFn,
    optimizer: torch.optim.Optimizer,
) -> float:
    net.train()
    total = 0.0
    for batch in loader:
        loss = step(batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def _eval_epoch(net: nn.Module, loader: Any, step: StepFn) -> float:
    net.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += step(batch).item()
    return total / max(len(loader), 1)


def fit(
    net: nn.Module,
    train_loader: Any,
    val_loader: Any,
    *,
    step: StepFn,
    n_epochs: int,
    lr: float,
    patience: int,
    min_lr: float = 0.0,
    verbose: bool = False,
    label: str = "",
) -> tuple[list[float], list[float]]:
    """Trains `net` with early stopping and restores its best weights.

    The same `step` computes the loss for training and validation batches,
    so anything that must differ between the two (noise augmentation)
    has to key off `net.training`.

    Args:
        net: The network; it is left in eval mode with the weights of the
            epoch with the lowest validation loss.
        train_loader: Batches of the training split.
        val_loader: Batches of the validation split.
        step: Maps a batch to its scalar loss.
        n_epochs: Maximum number of epochs.
        lr: Initial learning rate.
        patience: Epochs without improvement before stopping.
        min_lr: Lower bound of the plateau scheduler.
        verbose: Whether to print progress.
        label: Prefix for progress lines.

    Returns:
        (train_loss_history, val_loss_history), one value per epoch run.
    """
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=min_lr)
    train_hist: list[float] = []
    val_hist: list[float] = []
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(1, n_epochs + 1):
        avg_train = _train_epoch(net, train_loader, step, optimizer)
        avg_val = _eval_epoch(net, val_loader, step)
        scheduler.step(avg_val)
        train_hist.append(avg_train)
        val_hist.append(avg_val)

        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_state = copy.deepcopy(net.state_dict())
            stale = 0
        else:
            stale += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [{label}] Epoch {epoch:3d}/{n_epochs}  "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"best={best_val:.4f}  pat={stale}/{patience}")
        if stale >= patience:
            if verbose:
                print(f"  [{label}] Early stop at epoch {epoch}.")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return train_hist, val_hist
