"""LSTMPredictor: input scaling, physics scalars and legacy checkpoints.

predict() standardises each window with the training mean/std; until the
scaling test existed the training loop fed raw windows, so every LSTM-based
model was trained on one input distribution and evaluated on another.
"""
import io

import numpy as np
import pytest
import torch

from src.data.dataset import Dataset
from src.models.lstm import R2_SCALE_FLOOR, LSTMPredictor
from src.models.residual_net import ResidualNet

pytestmark = pytest.mark.usefixtures("single_thread")


def _toy_dataset(n=60, L=20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L) * 0.1
    g = rng.uniform(0.1, 0.5, n)
    coh = np.exp(-g[:, None] * t[None, :])
    seq = np.stack([
        3.0 + rng.normal(0, 0.1, (n, L)),          # deliberately off-centre
        -2.0 + rng.normal(0, 0.1, (n, L)),
        5.0 + rng.normal(0, 0.1, (n, L)),
        0.8 + 0.01 * rng.normal(size=(n, L)),       # purity
        coh,                                        # coherence_l1 (last)
    ], axis=-1).astype(np.float32)
    t_obs = np.full(n, t[-1] + 2.0, dtype=np.float32)
    t2 = (t_obs + 1.0 / g).astype(np.float32)
    rem = t2 - t_obs
    idx = rng.permutation(n)
    return Dataset(
        sequences=seq, t_obs=t_obs, t_decoh_abs=t2, remaining_time=rem,
        risk_labels=(rem <= 1.0).astype(np.float32),
        train_idx=idx[:40], val_idx=idx[40:50], test_idx=idx[50:],
        feature_names=["sigma_x", "sigma_y", "sigma_z", "purity",
                       "coherence_l1"],
        metadata={"dt": 0.1, "horizon": 1.0},
    )


def test_train_standardises_windows_like_predict():
    ds = _toy_dataset()
    seen = []
    p = LSTMPredictor(use_physics_prior=False)

    orig = torch.nn.LSTM.forward

    def spy(self, x, *a, **k):
        seen.append(x.detach().clone())
        return orig(self, x, *a, **k)

    torch.nn.LSTM.forward = spy
    try:
        p.train(ds, n_epochs=1, batch_size=64, lr=1e-3, seed=0, verbose=False)
        n_train_calls = len(seen)
        p.predict(ds.sequences[ds.train_idx[0]], float(ds.t_obs[0]), 1.0)
    finally:
        torch.nn.LSTM.forward = orig

    train_x = torch.cat(seen[:1])                 # first training batch
    infer_x = seen[n_train_calls]                 # the predict() call
    # Standardised inputs: channel means near 0 in both paths.
    assert train_x.mean(dim=(0, 1)).abs().max() < 0.5
    assert infer_x.mean(dim=(0, 1)).abs().max() < 1.5


def test_train_floors_the_scale_of_a_near_constant_fit_quality_scalar():
    """R²_OLS is ~constant on clean single-qubit windows; its scale must be
    floored so that measurement noise does not produce inputs of order 100."""
    ds = _toy_dataset()
    p = LSTMPredictor(use_physics_prior=True)
    p.train(ds, n_epochs=1, batch_size=64, lr=1e-3, seed=0, verbose=False)
    assert p.scalers["r2"].std >= R2_SCALE_FLOOR


def test_default_predictor_has_seven_physics_scalars():
    assert LSTMPredictor().use_J_scalar is False


def test_from_bytes_checkpoint_with_four_physics_scalars_still_predicts():
    """Checkpoints written before the extra scalars were added must load."""
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
