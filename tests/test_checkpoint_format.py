"""Checkpoint keys are the on-disk format; refactors must not change them."""
from __future__ import annotations

import io

import torch

from src.models import registry
from src.training import trainer
from tests import toy_data

LSTM_KEYS = [
    "state_dict", "feature_mean", "feature_std",
    "t_obs_mean", "t_obs_std", "slope_mean", "slope_std",
    "logc_mean", "logc_std", "phys_mean", "phys_std",
    "r2_mean", "r2_std", "ic_mean", "ic_std", "dc_mean", "dc_std",
    "residual_mean", "residual_std", "dt", "hidden_size", "num_layers",
    "dropout", "n_features", "use_physics_prior",
    "adaptive_prior_r2_threshold", "risk_pos_weight", "augment_sigma",
    "residual_output", "inference_interaction_code",
    "inference_dissipator_code", "use_J_scalar", "inference_J",
    "j_mean", "j_std",
]
TRANSFORMER_KEYS = [
    "state_dict", "feature_mean", "feature_std", "target_mean",
    "target_std", "dt", "d_model", "n_heads", "n_layers", "dim_ff",
    "dropout", "n_features",
]


def _keys(model) -> list[str]:
    ckpt = torch.load(io.BytesIO(model.state_bytes()), weights_only=False)
    return list(ckpt)


def _trained(alias: str, **overrides):
    model = registry.build_model(alias, **overrides)
    return trainer.train(model, toy_data.toy_dataset(), epochs=1, batch_size=8,
                         lr=1e-3, seed=0)


def test_lstm_checkpoint_keys_are_unchanged():
    assert _keys(_trained("physics_lstm", hidden_size=8)) == LSTM_KEYS


def test_transformer_checkpoint_keys_are_unchanged():
    model = _trained("transformer", d_model=16, n_heads=2, n_layers=1,
                     dim_ff=32)
    assert _keys(model) == TRANSFORMER_KEYS


def test_roundtrip_predicts_identically():
    model = _trained("physics_lstm", hidden_size=8)
    restored = type(model).from_bytes(model.state_bytes())
    ds = toy_data.toy_dataset()
    a = model.predict(ds.sequences[0], float(ds.t_obs[0]), 1.0)
    b = restored.predict(ds.sequences[0], float(ds.t_obs[0]), 1.0)
    assert a.t_decoh_predicted == b.t_decoh_predicted
