"""R15 — measurement-noise robustness of every model, across seeds.

Why
---
E3 measured only the residual model, in one run. With the BiLSTM training
fixed, the residual model degrades sharply under noise, and a diagnostic
shows the degradation comes from the physics scalars computed on the noisy
window (slope of log C, envelope switch, OLS R^2), not from the network:
fed the clean-window scalars, the same model is unaffected. To state this
as a comparison rather than a single curve, this probe trains each model on
clean data (E3 configuration: scenarios B+C, 100 trajectories each, window
50) and evaluates it on the test windows with Gaussian noise added to every
channel. The same noise realisation, scaled by sigma, is used at every
level, so the comparison across sigma is paired.

Writes ONLY to experiments/results/r15_noise_variants.csv.

Usage
-----
    python -m experiments.r15_noise_variants --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.run_noise_sweep import NoiseSweepUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS = Path(__file__).parent / "results" / "r15_noise_variants.csv"
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.10]


def _build(name):
    return {
        "physics_only": lambda: PhysicsOnlyPredictor(t_max=20.0),
        "lstm_only": lambda: LSTMPredictor(use_physics_prior=False),
        "physics_lstm": lambda: LSTMPredictor(use_physics_prior=True),
        "transformer": lambda: TransformerPredictor(),
    }[name]()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    n_per, epochs = (30, 10) if a.fast else (100, 100)
    gen = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
    with RESULTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "variant", "sigma", "r2", "mae", "auroc"])
        w.writeheader()
        for seed in a.seeds:
            groups = build_configs(n_per_scenario=n_per, seed=seed)
            ds = gen.execute(GenerateDatasetCommand(
                configs=groups["B"] + groups["C"], window_length=50, horizon=1.0,
                samples_per_trajectory=5, seed=seed, dataset_name=f"r15_{seed}"))
            for name in ("physics_only", "lstm_only", "physics_lstm", "transformer"):
                pred = _build(name)
                if name != "physics_only":
                    TrainModelUseCase(pred).execute(TrainModelCommand(
                        dataset=ds, n_epochs=epochs, batch_size=32, learning_rate=1e-3,
                        regression_loss="huber", verbose=False, seed=seed))
                for sigma in SIGMAS:
                    noisy = NoiseSweepUseCase._add_noise(ds, sigma, seed=seed)
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=noisy, horizon=1.0))
                    w.writerow({"seed": seed, "variant": name, "sigma": sigma,
                                "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc})
                    f.flush()
                    print(f"R15 | seed={seed} {name:<13} sigma={sigma:.2f} R2={m.r2:.3f} "
                          f"AUROC={m.risk_auroc:.3f}", flush=True)


if __name__ == "__main__":
    main()
