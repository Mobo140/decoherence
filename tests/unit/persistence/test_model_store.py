"""PickleModelStore and infer_predictor_kind: kind sidecar and legacy files."""
from __future__ import annotations

import numpy as np

from src.models.physics_only import PhysicsOnlyPredictor
from src.models.stretched_exp import StretchedExpPredictor
from src.models.transformer import TransformerNet, TransformerPredictor
from src.persistence.stores import PickleModelStore, infer_predictor_kind
from src.training.standardize import Standardizer


def _untrained_transformer() -> TransformerPredictor:
    p = TransformerPredictor(
        d_model=32, n_heads=4, n_layers=1, dim_ff=64, dropout=0.0)
    p.net = TransformerNet(
        n_features=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64, dropout=0.0,
    )
    p.feature_scaler = Standardizer(
        np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32))
    p.target_scaler = Standardizer(0.0, 1.0)
    p.dt = 0.1
    return p


def test_save_load_roundtrip_restores_physics_and_stretched_predictors(
        tmp_path):
    store = PickleModelStore(root=tmp_path)
    store.save(PhysicsOnlyPredictor(t_max=12.0, dt=0.2, risk_k=4.0), "phys")
    store.save(StretchedExpPredictor(t_max=9.0, dt=0.05), "sexp")

    phys = store.load("phys")
    sexp = store.load("sexp")
    assert isinstance(phys, PhysicsOnlyPredictor)
    assert phys.t_max == 12.0
    assert isinstance(sexp, StretchedExpPredictor)
    assert sexp.t_max == 9.0
    assert (tmp_path / "phys_kind.json").exists()


def test_infer_predictor_kind_reads_the_class_from_the_bytes():
    phys = PhysicsOnlyPredictor(t_max=7.0).state_bytes()
    sexp = StretchedExpPredictor(t_max=5.0).state_bytes()
    assert infer_predictor_kind(phys) == "PhysicsOnlyPredictor"
    assert infer_predictor_kind(sexp) == "StretchedExpPredictor"


def test_load_checkpoint_without_kind_file_infers_the_kind(tmp_path):
    store = PickleModelStore(root=tmp_path)
    pred = PhysicsOnlyPredictor(t_max=11.0)
    (tmp_path / "legacy.pt").write_bytes(pred.state_bytes())
    loaded = store.load("legacy")
    assert isinstance(loaded, PhysicsOnlyPredictor)
    assert loaded.t_max == 11.0


def test_save_load_roundtrip_restores_a_transformer(tmp_path):
    pred = _untrained_transformer()
    store = PickleModelStore(root=tmp_path)
    store.save(pred, "tf")
    loaded = store.load("tf")
    assert isinstance(loaded, TransformerPredictor)
    assert loaded.d_model == 32
    assert (tmp_path / "tf_kind.json").exists()


def test_load_transformer_without_kind_sidecar_infers_the_kind(tmp_path):
    pred = _untrained_transformer()
    (tmp_path / "legacy_tf.pt").write_bytes(pred.state_bytes())
    loaded = PickleModelStore(root=tmp_path).load("legacy_tf")
    assert isinstance(loaded, TransformerPredictor)
    assert infer_predictor_kind(pred.state_bytes()) == "TransformerPredictor"
