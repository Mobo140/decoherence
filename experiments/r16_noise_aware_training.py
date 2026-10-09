"""R16 — can noise-aware training make the residual model robust to noise?

Why
---
R15 showed that the residual model, best on clean data, collapses under
measurement noise (R^2 0.83 -> 0.31 at sigma = 0.05), because the slope of
log C over a short window and hence T_phys = -1/s - t_obs become unreliable,
while networks that do not use the estimate stay robust. A check without
training showed that smoothing the window (weighted or nonlinear fits, SSA,
exponential smoothing) does not fix the estimate itself.

During training the physics scalars are computed from clean windows only,
so the network never sees how far it can trust T_phys when the window is
noisy (the in-loop augmentation adds noise after the scalars are computed).
Here the training and validation sets are extended with noisy copies of
every window, and all inputs -- the window and the physics scalars derived
from it (slope, OLS R^2, T_phys) -- are computed from the noisy copy. The
standard error of the slope is a function of the slope and the OLS R^2, so
the network already has it.

Setup as in R15 (scenarios B+C, 100 trajectories each, window 50), three
seeds. Arms:
    physics_lstm        clean training (control, = R15)
    physics_lstm_aug    noise-aware training
    lstm_only_aug       noise-aware training, no physics estimate
    transformer         clean training
    transformer_aug     noise-aware training
Each training window gets two noisy copies with sigma drawn log-uniformly
from [0.005, 0.12]. Test windows are clean and get the same noise as in R15
(one realisation scaled by sigma, sigma in {0, 0.01, 0.02, 0.05, 0.10}).

Writes ONLY to experiments/results/r16_noise_aware_training.csv.

Usage
-----
    python -m experiments.r16_noise_aware_training --fast
    python -m experiments.r16_noise_aware_training --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.run_noise_sweep import NoiseSweepUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.quantum.noise import ar1_filter
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS = Path(__file__).parent / "results" / "r16_noise_aware_training.csv"
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.10]
ARMS = {
    "physics_lstm": (lambda: LSTMPredictor(use_physics_prior=True), False),
    "physics_lstm_aug": (lambda: LSTMPredictor(use_physics_prior=True), True),
    "lstm_only_aug": (lambda: LSTMPredictor(use_physics_prior=False), True),
    "transformer": (lambda: TransformerPredictor(), False),
    "transformer_aug": (lambda: TransformerPredictor(), True),
}


def noise_augmented(ds, copies: int, rng: np.random.Generator, phi_max: float = 0.0):
    """Append `copies` noisy versions of every train/val window.

    The noise goes into the raw window, so every quantity the model derives
    from it (log channels, slope, OLS R^2, T_phys) is computed from the noisy
    copy. Test windows are left untouched.

    phi_max > 0 makes the noise AR(1) in time with a lag-one correlation drawn
    per window from U[0, phi_max] (R22); the default draws exactly what R16
    drew.
    """
    per_window = [
        "sequences", "t_obs", "t_decoh_abs", "remaining_time", "risk_labels",
        "interaction_type_codes", "dissipator_type_codes", "J_values", "censored",
    ]
    n = len(ds.sequences)
    src = np.concatenate([ds.train_idx, ds.val_idx])
    is_train = np.concatenate([np.ones(len(ds.train_idx), bool), np.zeros(len(ds.val_idx), bool)])
    parts = {k: [getattr(ds, k)] for k in per_window}
    new_train, new_val = [ds.train_idx], [ds.val_idx]
    offset = n
    for _ in range(copies):
        sig = np.exp(rng.uniform(np.log(0.005), np.log(0.12), len(src))).astype(np.float32)
        eta = rng.normal(0.0, 1.0, ds.sequences[src].shape).astype(np.float32)
        if phi_max > 0.0:
            eta = ar1_filter(eta, rng.uniform(0.0, phi_max, len(src)))
        noisy = ds.sequences[src] + eta * sig[:, None, None]
        for k in per_window:
            arr = getattr(ds, k)
            parts[k].append(noisy if k == "sequences" else (arr[src] if len(arr) == n else arr))
        new_idx = offset + np.arange(len(src))
        new_train.append(new_idx[is_train]); new_val.append(new_idx[~is_train])
        offset += len(src)
    fields = {k: np.concatenate(v) if len(getattr(ds, k)) == n else getattr(ds, k) for k, v in parts.items()}
    return dataclasses.replace(ds, **fields,
                               train_idx=np.concatenate(new_train),
                               val_idx=np.concatenate(new_val))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    n_per, epochs = (30, 5) if a.fast else (100, 100)
    gen = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
    with RESULTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "sigma", "r2", "mae", "auroc"])
        w.writeheader()
        for seed in a.seeds:
            groups = build_configs(n_per_scenario=n_per, seed=seed)
            ds = gen.execute(GenerateDatasetCommand(
                configs=groups["B"] + groups["C"], window_length=50, horizon=1.0,
                samples_per_trajectory=5, seed=seed, dataset_name=f"r16_{seed}"))
            ds_aug = noise_augmented(ds, a.copies, np.random.default_rng(seed + 7))
            for arm, (build, aug) in ARMS.items():
                pred = build()
                TrainModelUseCase(pred).execute(TrainModelCommand(
                    dataset=ds_aug if aug else ds, n_epochs=epochs, batch_size=32,
                    learning_rate=1e-3, regression_loss="huber", verbose=False, seed=seed))
                for sigma in SIGMAS:
                    noisy = NoiseSweepUseCase._add_noise(ds, sigma, seed=seed)
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=noisy, horizon=1.0))
                    w.writerow({"seed": seed, "arm": arm, "sigma": sigma,
                                "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc})
                    f.flush()
                    print(f"R16 | seed={seed} {arm:<17} sigma={sigma:.2f} R2={m.r2:.3f} "
                          f"AUROC={m.risk_auroc:.3f}", flush=True)


if __name__ == "__main__":
    main()
