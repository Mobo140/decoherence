"""The BiLSTM must see the same input scaling at training and at inference.

predict() standardises each window with the training mean/std; until this
test existed the training loop fed raw windows, so every LSTM-based model was
trained on one input distribution and evaluated on another.
"""
import numpy as np
import torch

from src.application.generate_dataset import Dataset
from src.infrastructure.ml.lstm_predictor import LSTMPredictor


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
        feature_names=["sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"],
        metadata={"dt": 0.1, "horizon": 1.0},
    )


def test_training_windows_are_standardised_like_inference():
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


def test_near_constant_fit_quality_scalar_is_not_blown_up():
    """R²_OLS is ~constant on clean single-qubit windows; its scale must be
    floored so that measurement noise does not produce inputs of order 100."""
    from src.infrastructure.ml.training import R2_SCALE_FLOOR
    ds = _toy_dataset()
    p = LSTMPredictor(use_physics_prior=True)
    p.train(ds, n_epochs=1, batch_size=64, lr=1e-3, seed=0, verbose=False)
    assert p._r2_std >= R2_SCALE_FLOOR
