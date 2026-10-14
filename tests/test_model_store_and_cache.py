"""Audit leftovers: predictor kind on load, trajectory cache, E7a variant."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
    trajectory_cache_key,
)
from src.application.run_cross_system import variant_of
from src.domain.value_objects import (
    DissipatorConfig,
    PredictorVariant,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)
from src.models.physics_only import PhysicsOnlyPredictor
from src.models.stretched_exp import StretchedExpPredictor
from src.persistence.stores import (
    NumpyTrajectoryStore,
    PickleModelStore,
    infer_predictor_kind,
)


def _cfg() -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma=0.4),
        t_max=8.0,
        dt=0.2,
    )


def _traj(cfg: SystemConfig, censored: bool = False) -> TrajectoryResult:
    n = int(round(cfg.t_max / cfg.dt)) + 1
    times = np.linspace(0.0, cfg.t_max, n)
    return TrajectoryResult(
        times=times,
        observables={
            "sigma_x": np.ones(n, dtype=np.float32),
            "coherence_l1": np.exp(-times / 3.0).astype(np.float32),
            "purity": np.ones(n, dtype=np.float32),
        },
        t_decoh=3.0,
        system_config=cfg,
        censored=censored,
    )


class _CountingSim:
    def __init__(self, traj: TrajectoryResult) -> None:
        self.traj = traj
        self.calls = 0

    def simulate(self, config):
        self.calls += 1
        return self.traj


def test_lstm_from_bytes_old_4_physics_scalars():
    import io

    import torch

    from src.models.lstm import LSTMPredictor
    from src.models.residual_net import ResidualNet

    hidden, n_raw, n_in = 64, 5, 7
    net = ResidualNet(n_in, hidden, 1, 0.0, n_physics=4)
    buf = io.BytesIO()
    torch.save(
        {
            "state_dict": net.state_dict(),
            "feature_mean": np.zeros(n_in, dtype=np.float32),
            "feature_std": np.ones(n_in, dtype=np.float32),
            "t_obs_mean": 0.0,
            "t_obs_std": 1.0,
            "slope_mean": 0.0,
            "slope_std": 1.0,
            "logc_mean": 0.0,
            "logc_std": 1.0,
            "phys_mean": 0.0,
            "phys_std": 1.0,
            "residual_mean": 0.0,
            "residual_std": 1.0,
            "dt": 0.1,
            "hidden_size": hidden,
            "num_layers": 1,
            "dropout": 0.0,
            "n_features": n_in,
        },
        buf,
    )
    loaded = LSTMPredictor.from_bytes(buf.getvalue())
    assert loaded.net.n_physics == 4
    window = np.ones((12, n_raw), dtype=np.float32)
    window[:, -1] = np.linspace(1.0, 0.2, 12, dtype=np.float32)
    pred = loaded.predict(window, 1.0, 1.0)
    assert pred.t_decoh_predicted >= 0.0


def test_best_registry_swallows_bad_checkpoint(tmp_path):
    from src.persistence.stores import BestModelRegistry

    (tmp_path / "best_1qubit.pt").write_bytes(b"not-a-checkpoint")
    assert BestModelRegistry(root=tmp_path).get_predictor(1) is None


def test_store_roundtrip_physics_and_stretched(tmp_path):
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


def test_infer_kind_without_sidecar():
    phys = PhysicsOnlyPredictor(t_max=7.0).state_bytes()
    sexp = StretchedExpPredictor(t_max=5.0).state_bytes()
    assert infer_predictor_kind(phys) == "PhysicsOnlyPredictor"
    assert infer_predictor_kind(sexp) == "StretchedExpPredictor"


def test_load_old_checkpoint_without_kind_file(tmp_path):
    store = PickleModelStore(root=tmp_path)
    pred = PhysicsOnlyPredictor(t_max=11.0)
    (tmp_path / "legacy.pt").write_bytes(pred.state_bytes())
    loaded = store.load("legacy")
    assert isinstance(loaded, PhysicsOnlyPredictor)
    assert loaded.t_max == 11.0


def test_variant_of_predictor_classes():
    from src.models.lstm import LSTMPredictor
    from src.models.transformer import TransformerPredictor

    assert variant_of(TransformerPredictor()) == PredictorVariant.TRANSFORMER
    assert variant_of(PhysicsOnlyPredictor()) == PredictorVariant.PHYSICS_ONLY
    assert variant_of(StretchedExpPredictor()) == PredictorVariant.STRETCHED_EXP
    assert variant_of(LSTMPredictor(use_physics_prior=False)) == PredictorVariant.LSTM_ONLY
    assert variant_of(LSTMPredictor()) == PredictorVariant.PHYSICS_LSTM


def test_e7a_csv_variant_is_transformer():
    path = Path(__file__).resolve().parents[1] / "experiments" / "results" / "e7a_transformer_cross.csv"
    text = path.read_text()
    assert "transformer" in text
    assert "physics_lstm" not in text


def test_as_test_only_puts_every_window_in_test_idx():
    from src.data.dataset import Dataset

    n = 6
    ds = Dataset(
        sequences=np.zeros((n, 4, 3), dtype=np.float32),
        t_obs=np.ones(n, dtype=np.float32),
        t_decoh_abs=np.full(n, 5.0, dtype=np.float32),
        remaining_time=np.full(n, 4.0, dtype=np.float32),
        risk_labels=np.zeros(n, dtype=np.int64),
        train_idx=np.array([0, 1, 2], dtype=np.int64),
        val_idx=np.array([3], dtype=np.int64),
        test_idx=np.array([4, 5], dtype=np.int64),
        feature_names=["a", "b", "c"],
        metadata={},
    )
    out = ds.as_test_only()
    assert list(out.test_idx) == list(range(n))
    assert len(out.train_idx) == 0
    assert len(out.val_idx) == 0


def _untrained_transformer():
    from src.models.transformer import TransformerPredictor, TransformerNet
    from src.training.standardize import Standardizer

    p = TransformerPredictor(d_model=32, n_heads=4, n_layers=1, dim_ff=64, dropout=0.0)
    p.net = TransformerNet(
        n_features=5, d_model=32, n_heads=4, n_layers=1, dim_ff=64, dropout=0.0,
    )
    p.feature_scaler = Standardizer(
        np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32))
    p.target_scaler = Standardizer(0.0, 1.0)
    p.dt = 0.1
    return p


def test_store_roundtrip_transformer(tmp_path):
    from src.models.transformer import TransformerPredictor

    pred = _untrained_transformer()
    store = PickleModelStore(root=tmp_path)
    store.save(pred, "tf")
    loaded = store.load("tf")
    assert isinstance(loaded, TransformerPredictor)
    assert loaded.d_model == 32
    assert (tmp_path / "tf_kind.json").exists()


def test_load_transformer_without_kind_sidecar(tmp_path):
    from src.models.transformer import TransformerPredictor

    pred = _untrained_transformer()
    (tmp_path / "legacy_tf.pt").write_bytes(pred.state_bytes())
    loaded = PickleModelStore(root=tmp_path).load("legacy_tf")
    assert isinstance(loaded, TransformerPredictor)
    assert infer_predictor_kind(pred.state_bytes()) == "TransformerPredictor"


def test_trajectory_cache_skips_second_simulate(tmp_path):
    cfg = _cfg()
    traj = _traj(cfg, censored=False)
    sim = _CountingSim(traj)
    store = NumpyTrajectoryStore(root=tmp_path)
    uc = GenerateDatasetUseCase(sim, store=store)
    cmd = GenerateDatasetCommand(
        configs=[cfg],
        window_length=8,
        samples_per_trajectory=2,
        seed=7,
        dataset_name="bulk",
    )
    ds1 = uc.execute(cmd)
    assert sim.calls == 1
    assert len(ds1.sequences) > 0

    sim2 = _CountingSim(_traj(cfg, censored=True))
    uc2 = GenerateDatasetUseCase(sim2, store=store)
    ds2 = uc2.execute(cmd)
    assert sim2.calls == 0
    assert len(ds2.sequences) == len(ds1.sequences)
    key = trajectory_cache_key(cfg, 7, 0)
    assert (tmp_path / f"{key}.npz").exists()


def test_store_persists_censored_flag(tmp_path):
    cfg = _cfg()
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj(cfg, censored=True)], "one")
    loaded = store.load("one")
    assert loaded[0].censored is True
