"""The public training helper."""
from __future__ import annotations

import numpy as np
import torch

from src.data import dataset as dataset_lib
from src.models import registry
from src.training import trainer


def _dataset(n: int = 24, window: int = 10) -> dataset_lib.Dataset:
    rng = np.random.default_rng(0)
    t = np.arange(window) * 0.1
    rates = rng.uniform(0.3, 1.0, n)
    seq = np.exp(-rates[:, None] * t[None, :])[:, :, None]
    seq = np.concatenate([seq, np.ones_like(seq)], axis=-1).astype(np.float32)
    t_obs = np.full(n, t[-1], dtype=np.float32)
    remaining = (2.0 / rates - t_obs).astype(np.float32)
    idx = np.arange(n)
    return dataset_lib.Dataset(
        sequences=seq, t_obs=t_obs, t_decoh_abs=t_obs + remaining,
        remaining_time=remaining,
        risk_labels=(remaining <= 1.0).astype(np.float32),
        train_idx=idx[:16], val_idx=idx[16:20], test_idx=idx[20:],
        feature_names=["coherence_l1", "purity"], metadata={"dt": 0.1},
    )


def test_train_returns_the_same_model_trained():
    model = registry.build_model("physics_lstm", hidden_size=8, dropout=0.0)
    out = trainer.train(model, _dataset(), epochs=1, batch_size=8,
                        lr=1e-3, seed=3)
    assert out is model
    assert model.is_trained


def test_same_seed_gives_the_same_weights():
    ds = _dataset()
    weights = []
    for _ in range(2):
        model = registry.build_model("transformer", d_model=16, n_heads=2,
                                     n_layers=1, dim_ff=32)
        trainer.train(model, ds, epochs=1, batch_size=8, lr=1e-3, seed=5)
        weights.append([w.detach().clone() for w in model.net.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*weights))


def test_untrainable_model_passes_through():
    model = registry.build_model("physics_only")
    out = trainer.train(model, _dataset(), epochs=1, batch_size=8,
                        lr=1e-3, seed=0)
    assert out is model
