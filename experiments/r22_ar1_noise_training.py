"""R22 — noise-aware training with time-correlated (AR(1)) noise.

Why
---
R21 showed that measurement noise correlated in time hurts more than white
noise of the same size: at phi = 0.9 the variance of the OLS slope over a
50-point window grows 9.6-fold, every trained network loses accuracy, and
the residual model trained with white noise (R16) loses its lead over the
Transformer. The noisy copies in R16 were white, so the network never saw a
window in which a slow drift of the noise imitates a trend. Here the noisy
copies are AR(1) with a lag-one correlation drawn per window from
U[0, phi_max]; everything else is as in R16.

Questions
  * Does correlated training noise restore the residual model's lead at
    phi = 0.9?
  * What does it cost on white noise and on clean data?
  * Do the networks without the slope estimate gain as much?

Setup as in R16/R21 (scenarios B+C, 100 trajectories each, window 50).
Arms:
    physics_lstm_white  white noisy copies (= R16 physics_lstm_aug)
    physics_lstm_ar1    AR(1) noisy copies, phi ~ U[0, 0.95]
    lstm_only_ar1       same, no physics estimate
    transformer         clean training (= R21)
    transformer_ar1     AR(1) noisy copies
Each training window gets two noisy copies, sigma log-uniform in
[0.005, 0.12] as in R16. Test windows get AR(1) noise with sigma in
{0, 0.01, 0.02, 0.05, 0.10} and phi in {0, 0.5, 0.9}; one standard-normal
realisation per seed is filtered for every phi and scaled by sigma, so all
comparisons are paired and phi = 0 is the R15/R16 test noise.

Writes ONLY to experiments/results/r22_ar1_noise_training*.csv.

Usage
-----
    python -m experiments.r22_ar1_noise_training --fast
    python -m experiments.r22_ar1_noise_training --seeds 42 --tag s42
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs, trajectory_store
from experiments.r16_noise_aware_training import noise_augmented
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.infrastructure.quantum.noise import ar1_filter
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.10]
PHIS = [0.0, 0.5, 0.9]
PHI_MAX = 0.95

# name -> (builder, training set)
ARMS = {
    "physics_lstm_white": (lambda: LSTMPredictor(use_physics_prior=True), "white"),
    "physics_lstm_ar1": (lambda: LSTMPredictor(use_physics_prior=True), "ar1"),
    "lstm_only_ar1": (lambda: LSTMPredictor(use_physics_prior=False), "ar1"),
    "transformer": (lambda: TransformerPredictor(), "clean"),
    "transformer_ar1": (lambda: TransformerPredictor(), "ar1"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    n_per, epochs = (30, 5) if a.fast else (100, 100)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out = RESULTS_DIR / f"r22_ar1_noise_training{sfx}.csv"
    print(f"R22 | seeds={a.seeds} n_per={n_per} epochs={epochs}", flush=True)

    gen = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "sigma", "phi", "r2", "mae",
                                          "auroc", "n_samples"])
        w.writeheader()
        for seed in a.seeds:
            ts = time.time()
            groups = build_configs(n_per_scenario=n_per, seed=seed)
            ds = gen.execute(GenerateDatasetCommand(
                configs=groups["B"] + groups["C"], window_length=50, horizon=1.0,
                samples_per_trajectory=5, seed=seed, dataset_name=f"r22_{seed}"))
            # Same generator seed as R16/R21, so the white copies are R16's.
            train_sets = {
                "clean": ds,
                "white": noise_augmented(ds, 2, np.random.default_rng(seed + 7)),
                "ar1": noise_augmented(ds, 2, np.random.default_rng(seed + 7), phi_max=PHI_MAX),
            }
            eta = np.random.default_rng(seed).normal(0.0, 1.0, size=ds.sequences.shape)
            tests = {}
            for phi in PHIS:
                e = ar1_filter(eta, phi)
                for sigma in SIGMAS:
                    if sigma == 0.0 and phi > 0.0:
                        continue
                    t = copy.copy(ds)
                    t.sequences = (ds.sequences + sigma * e).astype(np.float32)
                    tests[(sigma, phi)] = t

            for arm, (build, train) in ARMS.items():
                pred = build()
                TrainModelUseCase(pred).execute(TrainModelCommand(
                    dataset=train_sets[train], n_epochs=epochs, batch_size=32,
                    learning_rate=1e-3, regression_loss="huber", verbose=False, seed=seed))
                for (sigma, phi), tds in tests.items():
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=tds, horizon=1.0))
                    w.writerow({"seed": seed, "arm": arm, "sigma": sigma, "phi": phi,
                                "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc,
                                "n_samples": m.n_samples})
                    print(f"R22 | seed={seed} {arm:<19} sigma={sigma:.2f} phi={phi:.1f} "
                          f"R2={m.r2:.3f}", flush=True)
                f.flush()
            print(f"R22 | seed={seed} done [{time.time() - ts:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
