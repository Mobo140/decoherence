"""The public training helper."""
from __future__ import annotations

import pytest
import torch

from src.models import registry
from src.training import trainer

pytestmark = pytest.mark.usefixtures("single_thread")


def test_train_returns_the_same_model_trained(toy_dataset):
    model = registry.build_model("physics_lstm", hidden_size=8, dropout=0.0)
    out = trainer.train(model, toy_dataset, epochs=1, batch_size=8,
                        lr=1e-3, seed=3)
    assert out is model
    assert model.is_trained


def test_train_with_the_same_seed_gives_the_same_weights(toy_dataset):
    ds = toy_dataset
    weights = []
    for _ in range(2):
        model = registry.build_model("transformer", d_model=16, n_heads=2,
                                     n_layers=1, dim_ff=32)
        trainer.train(model, ds, epochs=1, batch_size=8, lr=1e-3, seed=5)
        weights.append([w.detach().clone() for w in model.net.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*weights))


def test_train_untrainable_model_passes_through(toy_dataset):
    model = registry.build_model("physics_only")
    out = trainer.train(model, toy_dataset, epochs=1, batch_size=8,
                        lr=1e-3, seed=0)
    assert out is model
