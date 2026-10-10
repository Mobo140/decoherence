"""Batch construction helpers for LSTMPredictor training."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset




# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def add_log(X: np.ndarray, feature_names: list) -> np.ndarray:
    names = list(feature_names)
    ci = names.index("coherence_l1") if "coherence_l1" in names else X.shape[-1] - 1
    pi = names.index("purity")       if "purity"       in names else X.shape[-1] - 2
    lc = np.log(np.abs(X[..., ci:ci+1]) + 1e-8).astype(np.float32)
    lp = np.log(np.abs(X[..., pi:pi+1]) + 1e-8).astype(np.float32)
    return np.concatenate([X.astype(np.float32), lc, lp], axis=-1)


def scalar_tensor(val: float, mean: float, std: float, device: str) -> torch.Tensor:
    return torch.tensor([(val - mean) / std], dtype=torch.float32).to(device)


def make_loader(X, y, t, s, lc, ph, r2, ic, dc, j, r, rem, cens, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X,    dtype=torch.float32),
        torch.tensor(y,    dtype=torch.float32),
        torch.tensor(t,    dtype=torch.float32),
        torch.tensor(s,    dtype=torch.float32),
        torch.tensor(lc,   dtype=torch.float32),
        torch.tensor(ph,   dtype=torch.float32),
        torch.tensor(r2,   dtype=torch.float32),
        torch.tensor(ic,   dtype=torch.float32),
        torch.tensor(dc,   dtype=torch.float32),
        torch.tensor(j,    dtype=torch.float32),
        torch.tensor(r,    dtype=torch.float32),
        torch.tensor(rem,  dtype=torch.float32),
        torch.tensor(cens, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
