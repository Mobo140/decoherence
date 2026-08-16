"""Experiment E13 — Survival loss vs dropping censored TFIM trajectories.

Goal
----
Compare keep+SurvHuber against the drop-censored pipeline from task 06
on TFIM. SurvHuber should not hurt uncensored-test R² and should use
more windows when T₂ > t_max is common.

Claim
-----
C2: TFIM needs correct labels, not only ML. Right-censored T₂ must not
    be treated as t_max and should not be dropped without a survival loss.

Scenarios
---------
E (2q TFIM). window=30, hidden=128, τ=0, E8 regularisation.
drop: include_censored=False, Huber. surv: include_censored=True, SurvHuber.

Metrics
-------
R² on uncensored test windows only. At t_max=20 censoring is rare
(~5 windows / 200 trajs); no gain until γ/t_max produce more T₂ > t_max.

Output
------
experiments/results/e13_survival_tfim.csv

Usage
-----
    python -m experiments.e13_survival_tfim
    python -m experiments.e13_survival_tfim --fast
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"


def _lstm() -> LSTMPredictor:
    return LSTMPredictor(
        hidden_size=128, num_layers=2, dropout=0.3,
        use_physics_prior=True,
        adaptive_prior_r2_threshold=0.0,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=3.0,
        early_stopping_patience=30,
    )


def _run_arm(simulator, configs, *, include_censored: bool, loss: str,
             n_epochs: int, window: int, verbose: bool) -> dict:
    ds = GenerateDatasetUseCase(simulator, store=trajectory_store()).execute(GenerateDatasetCommand(
        configs=configs,
        window_length=window,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        min_window_gap=max(window // 2, 1),
        include_censored=include_censored,
    ))
    pred = _lstm()
    TrainModelUseCase(pred).execute(TrainModelCommand(
        dataset=ds,
        n_epochs=n_epochs,
        batch_size=64,
        learning_rate=5e-4,
        regression_loss=loss,
        verbose=verbose,
    ))
    metrics = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
    n_cens_win = int(ds.metadata.get("n_censored_windows", 0))
    print(f"  → {metrics.summary()}  censored_windows={n_cens_win}")
    return {
        "name": "surv" if include_censored else "drop",
        "variant": PredictorVariant.PHYSICS_LSTM.value,
        "scenario": "E",
        "include_censored": include_censored,
        "loss": loss,
        "mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2,
        "mape": metrics.mape, "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
        "n_censored_traj": ds.metadata.get("n_censored", 0),
        "n_censored_windows": n_cens_win,
        "n_train": ds.metadata.get("n_train", 0),
    }


def main(fast: bool = False) -> None:
    n_per = 40 if fast else 200
    epochs = 15 if fast else 80
    window = 20 if fast else 30
    groups = build_configs(n_per_scenario=n_per, seed=42)
    simulator = QuTipSimulator()

    print(f"E13 survival TFIM | n_per={n_per} | epochs={epochs} | window={window}")
    rows = []
    print("\n=== drop censored + Huber ===")
    rows.append(_run_arm(
        simulator, groups["E"], include_censored=False, loss="huber",
        n_epochs=epochs, window=window, verbose=True,
    ))
    print("\n=== keep censored + SurvHuber ===")
    rows.append(_run_arm(
        simulator, groups["E"], include_censored=True, loss="surv_huber",
        n_epochs=epochs, window=window, verbose=True,
    ))

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "e13_survival_tfim.csv"
    fields = [
        "name", "variant", "scenario", "include_censored", "loss",
        "mae", "rmse", "r2", "mape", "auroc", "n_samples",
        "n_censored_traj", "n_censored_windows", "n_train",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {path}")
    print(f"\n{'arm':<8} {'R²':>8} {'n_test':>8} {'n_train':>8} {'cens_win':>8}")
    for r in rows:
        print(f"{r['name']:<8} {r['r2']:>8.4f} {r['n_samples']:>8} "
              f"{r['n_train']:>8} {r['n_censored_windows']:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
