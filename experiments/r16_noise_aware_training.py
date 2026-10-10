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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.data import observation
from src.data.scenarios import build_configs, trajectory_store
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.simulation.simulator import QuTipSimulator

RESULTS = Path(__file__).parent / "results" / "r16_noise_aware_training.csv"
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.10]
ARMS = {
    "physics_lstm": (lambda: LSTMPredictor(use_physics_prior=True), False),
    "physics_lstm_aug": (lambda: LSTMPredictor(use_physics_prior=True), True),
    "lstm_only_aug": (lambda: LSTMPredictor(use_physics_prior=False), True),
    "transformer": (lambda: TransformerPredictor(), False),
    "transformer_aug": (lambda: TransformerPredictor(), True),
}


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
            ds_aug = observation.noise_augmented(ds, a.copies, np.random.default_rng(seed + 7))
            for arm, (build, aug) in ARMS.items():
                pred = build()
                TrainModelUseCase(pred).execute(TrainModelCommand(
                    dataset=ds_aug if aug else ds, n_epochs=epochs, batch_size=32,
                    learning_rate=1e-3, regression_loss="huber", verbose=False, seed=seed))
                for sigma in SIGMAS:
                    noisy = observation.OBSERVATIONS["gaussian"](
                        ds, sigma=sigma, rng=np.random.default_rng(seed))
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=noisy, horizon=1.0))
                    w.writerow({"seed": seed, "arm": arm, "sigma": sigma,
                                "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc})
                    f.flush()
                    print(f"R16 | seed={seed} {arm:<17} sigma={sigma:.2f} R2={m.r2:.3f} "
                          f"AUROC={m.risk_auroc:.3f}", flush=True)


if __name__ == "__main__":
    main()
