"""Shared training loop and standardiser."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils import data as torch_data

from src.training import loop
from src.training import standardize


def test_standardizer_scalar_uses_float_statistics_with_eps():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = standardize.Standardizer.fit(x)
    assert isinstance(s.mean, float) and isinstance(s.std, float)
    assert s.mean == pytest.approx(2.5)
    assert s.std == pytest.approx(float(x.std()) + standardize.EPS)
    assert s.apply(2.5) == pytest.approx(0.0)


def test_standardizer_keeps_channel_axis_for_windows():
    x = np.random.default_rng(0).normal(size=(5, 7, 3)).astype(np.float32)
    s = standardize.Standardizer.fit(x, axis=(0, 1))
    assert s.mean.shape == (3,)
    z = s.apply(x)
    assert np.allclose(z.mean(axis=(0, 1)), 0.0, atol=1e-5)


def test_standardizer_min_std_replaces_eps():
    s = standardize.Standardizer.fit(np.full(10, 0.5), min_std=0.05)
    assert s.std == 0.05


def test_default_standardizer_is_identity_on_tensors():
    t = torch.tensor([1.0, 2.0])
    assert torch.equal(standardize.Standardizer().apply(t), t)


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


def test_loop_stops_after_patience_and_restores_best_weights():
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


def test_loop_runs_all_epochs_while_improving():
    per_epoch = [1.0, 0.9, 0.8, 0.7]
    scripted = [v for v in per_epoch for _ in range(2)]
    _, (train_hist, val_hist), _ = _run(scripted, patience=2)
    assert len(val_hist) == 4
