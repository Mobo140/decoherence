"""Shared training loop: early stopping, best-weight restore, options."""
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils import data as torch_data

from src.training import loop

pytestmark = pytest.mark.usefixtures("single_thread")


def test_fit_options_ignore_unknown_keys():
    options = loop.FitOptions.from_kwargs(
        {"n_epochs": 3, "seed": 4, "something_else": 1})
    assert (options.n_epochs, options.seed, options.batch_size) == (3, 4, 32)


def _toy_loader():
    x = torch.linspace(-1, 1, 8).unsqueeze(1)
    return torch_data.DataLoader(
        torch_data.TensorDataset(x, 2 * x), batch_size=4, shuffle=False)


def _run(val_losses, patience):
    """Runs the loop with a scripted validation loss per epoch."""
    torch.manual_seed(0)
    net = nn.Linear(1, 1)
    scripted = iter(val_losses)
    weights_at_epoch = []

    def step(batch):
        x, y = batch
        if net.training:
            return ((net(x) - y) ** 2).mean()
        return torch.tensor(next(scripted))

    # The toy loader has two batches, so each epoch consumes two scripted
    # values and its mean is the scripted loss.
    def recording_step(batch):
        loss = step(batch)
        if not net.training:
            weights_at_epoch.append(net.weight.detach().clone())
        return loss

    hist = loop.fit(
        net, _toy_loader(), _toy_loader(), step=recording_step,
        n_epochs=len(val_losses) // 2, lr=1e-2, patience=patience)
    return net, hist, weights_at_epoch


def test_fit_stops_after_patience_and_restores_best_weights():
    # Epoch 2 is best; epochs 3-5 do not improve by MIN_DELTA, so with
    # patience 3 the loop stops after epoch 5 of 8.
    per_epoch = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9]
    scripted = [v for v in per_epoch for _ in range(2)]
    net, (train_hist, val_hist), seen = _run(scripted, patience=3)
    assert len(train_hist) == len(val_hist) == 5
    assert val_hist[:2] == pytest.approx([1.0, 0.5])
    # Two validation batches per epoch were recorded; the best epoch is 2.
    assert torch.equal(net.weight, seen[3])
    assert not net.training


def test_fit_runs_all_epochs_while_improving():
    per_epoch = [1.0, 0.9, 0.8, 0.7]
    scripted = [v for v in per_epoch for _ in range(2)]
    _, (train_hist, val_hist), _ = _run(scripted, patience=2)
    assert len(val_hist) == 4
