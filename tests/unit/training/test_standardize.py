"""Standardizer: fitted statistics, channel axes and the identity default."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training import standardize


def test_fit_scalar_uses_float_statistics_with_eps():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = standardize.Standardizer.fit(x)
    assert isinstance(s.mean, float) and isinstance(s.std, float)
    assert s.mean == pytest.approx(2.5)
    assert s.std == pytest.approx(float(x.std()) + standardize.EPS)
    assert s.apply(2.5) == pytest.approx(0.0)


def test_fit_over_windows_keeps_the_channel_axis():
    x = np.random.default_rng(0).normal(size=(5, 7, 3)).astype(np.float32)
    s = standardize.Standardizer.fit(x, axis=(0, 1))
    assert s.mean.shape == (3,)
    z = s.apply(x)
    assert np.allclose(z.mean(axis=(0, 1)), 0.0, atol=1e-5)


def test_fit_min_std_replaces_eps():
    s = standardize.Standardizer.fit(np.full(10, 0.5), min_std=0.05)
    assert s.std == 0.05


def test_default_standardizer_is_identity_on_tensors():
    t = torch.tensor([1.0, 2.0])
    assert torch.equal(standardize.Standardizer().apply(t), t)
