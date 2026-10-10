"""The public training helper."""
from __future__ import annotations

import torch

from src.models import registry
from src.training import trainer
from tests import toy_data


def test_train_returns_the_same_model_trained():
    model = registry.build_model("physics_lstm", hidden_size=8, dropout=0.0)
    out = trainer.train(model, toy_data.toy_dataset(), epochs=1, batch_size=8,
                        lr=1e-3, seed=3)
    assert out is model
    assert model.is_trained


def test_same_seed_gives_the_same_weights():
    ds = toy_data.toy_dataset()
    weights = []
    for _ in range(2):
        model = registry.build_model("transformer", d_model=16, n_heads=2,
                                     n_layers=1, dim_ff=32)
        trainer.train(model, ds, epochs=1, batch_size=8, lr=1e-3, seed=5)
        weights.append([w.detach().clone() for w in model.net.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*weights))


def test_untrainable_model_passes_through():
    model = registry.build_model("physics_only")
    out = trainer.train(model, toy_data.toy_dataset(), epochs=1, batch_size=8,
                        lr=1e-3, seed=0)
    assert out is model
