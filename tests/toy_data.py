"""A tiny exponential-decay dataset for training tests."""
from __future__ import annotations

import numpy as np

from src.data import dataset as dataset_lib


def toy_dataset(n: int = 24, window: int = 10) -> dataset_lib.Dataset:
    """Builds windows of exact exponential decay with random rates.

    Args:
        n: Number of windows; 16 train, 4 validation, the rest test.
        window: Window length.

    Returns:
        A dataset with the features coherence_l1 and purity.
    """
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
