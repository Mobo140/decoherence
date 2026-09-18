"""Experiment E10 (fixed-context) — Corrected codes + Transformer cross-system.

Goal
----
Re-run context injection with the 4-class codes that match the dataset
encoder, and try Transformer with E8c training settings.

Claim
-----
C5: correct system codes at train and eval raise R² vs the E9 buggy codes
    (D/E were sent as 0/1). Transformer + E8c window/gap should beat E7a.

Scenarios
---------
D (code 2), E (code 3). Do not overwrite e10_tfim_*.csv from the later E10.

Metrics
-------
R², MAE, AUROC vs E8c (D=0.740, E=0.575) and E7a (D=0.758, E=0.599).

Output
------
experiments/results/e10a_cross_system.csv
experiments/results/e10b_cross_system.csv

Usage
-----
    python -m experiments.e10_fixed_context
    python -m experiments.e10_fixed_context --fast
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.domain.value_objects import (
    AblationResult, ExperimentSpec, NoiseConfig, PredictorVariant, SystemConfig,
)
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"
WINDOW_2Q   = 30
MIN_GAP     = 15

# 4-class system-type codes (must match generate_dataset.py)
SCENARIO_CODE = {"D": 2.0, "E": 3.0}


def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def _force_all_test(dataset):
    ds = copy.copy(dataset)
    ds.test_idx  = np.arange(len(dataset.sequences))
    ds.train_idx = np.array([], dtype=np.int64)
    ds.val_idx   = np.array([], dtype=np.int64)
    return ds


# ---------------------------------------------------------------------------
# E10a — physics_lstm, fixed 4-class context codes
# ---------------------------------------------------------------------------

def run_e10a(simulator, groups, n_epochs: int, n_per: int, verbose: bool = True) -> list:
    """Cross-system physics_lstm with corrected system-type codes."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    print("=== E10a: training physics_lstm on D+E mixture (fixed 4-class codes) ===")

    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())

    train_dataset = gen_uc.execute(GenerateDatasetCommand(
        configs=groups["D"] + groups["E"],
        window_length=WINDOW_2Q,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        train_ratio=0.9,
        val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))

    predictor = LSTMPredictor(
        hidden_size=128, num_layers=2, dropout=0.3,
        use_physics_prior=True,
        adaptive_prior_r2_threshold=0.5,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=2.0,  # XXZ default; overridden per scenario below
        inference_dissipator_code=0.0,
    )
    TrainModelUseCase(predictor).execute(TrainModelCommand(
        dataset=train_dataset,
        n_epochs=n_epochs,
        batch_size=64,
        learning_rate=5e-4,
        regression_loss="huber",
        verbose=verbose,
        seed=42,
    ))

    rows = []
    for scenario_label in ["D", "E"]:
        # Set the correct inference code for this scenario
        predictor.inference_interaction_code = SCENARIO_CODE[scenario_label]

        eval_dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=groups[scenario_label],
            window_length=WINDOW_2Q,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=43,
            train_ratio=0.0,
            val_ratio=0.0,
            min_window_gap=MIN_GAP,
        ))
        eval_dataset = _force_all_test(eval_dataset)

        metrics = BacktestUseCase(predictor).execute(
            BacktestCommand(dataset=eval_dataset, horizon=1.0)
        )
        print(f"  [{scenario_label}] → {metrics.summary()}")

        rows.append({
            "name": f"cross_system_{scenario_label}",
            "variant": "physics_lstm",
            "window_fraction": WINDOW_2Q,
            "noise_sigma": 0.0,
            "mae":  metrics.mae,
            "rmse": metrics.rmse,
            "r2":   metrics.r2,
            "mape": metrics.mape,
            "auroc": metrics.risk_auroc,
            "n_samples": metrics.n_samples,
        })

    _save_csv(rows, RESULTS_DIR / "e10a_cross_system.csv",
              ["name", "variant", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return rows


# ---------------------------------------------------------------------------
# E10b — Transformer cross-system, E8c training settings
# ---------------------------------------------------------------------------

def run_e10b(simulator, groups, n_epochs: int, n_per: int, verbose: bool = True) -> list:
    """Cross-system Transformer with E8c config."""
    from src.infrastructure.ml.transformer_predictor import TransformerPredictor

    print("=== E10b: training Transformer on D+E mixture (E8c config) ===")

    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())

    train_dataset = gen_uc.execute(GenerateDatasetCommand(
        configs=groups["D"] + groups["E"],
        window_length=WINDOW_2Q,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        train_ratio=0.9,
        val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))

    predictor = TransformerPredictor(
        d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1,
    )
    TrainModelUseCase(predictor).execute(TrainModelCommand(
        dataset=train_dataset,
        n_epochs=n_epochs,
        batch_size=64,
        learning_rate=5e-4,
        regression_loss="huber",
        verbose=verbose,
        seed=42,
    ))

    rows = []
    for scenario_label in ["D", "E"]:
        eval_dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=groups[scenario_label],
            window_length=WINDOW_2Q,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=43,
            train_ratio=0.0,
            val_ratio=0.0,
            min_window_gap=MIN_GAP,
        ))
        eval_dataset = _force_all_test(eval_dataset)

        metrics = BacktestUseCase(predictor).execute(
            BacktestCommand(dataset=eval_dataset, horizon=1.0)
        )
        print(f"  [{scenario_label}] → {metrics.summary()}")

        rows.append({
            "name": f"cross_system_{scenario_label}",
            "variant": "transformer",
            "window_fraction": WINDOW_2Q,
            "noise_sigma": 0.0,
            "mae":  metrics.mae,
            "rmse": metrics.rmse,
            "r2":   metrics.r2,
            "mape": metrics.mape,
            "auroc": metrics.risk_auroc,
            "n_samples": metrics.n_samples,
        })

    _save_csv(rows, RESULTS_DIR / "e10b_cross_system.csv",
              ["name", "variant", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_per  = 50  if fast else 500
    epochs = 10  if fast else 150

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'='*70}")
    print(f"E10a — Fixed context (physics_lstm) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    rows_a = run_e10a(simulator, groups, n_epochs=epochs, n_per=n_per)

    print(f"\n{'='*70}")
    print(f"E10b — Transformer cross-system | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    rows_b = run_e10b(simulator, groups, n_epochs=epochs, n_per=n_per)

    print("\n\n=== SUMMARY ===")
    print(f"\n{'Model':<20} {'Scenario':<10} {'R²':>8} {'AUROC':>8}")
    print("-" * 50)
    baselines = {
        "D": {"physics_lstm E8c": (0.740, 0.789), "Transformer E7a": (0.758, 0.863)},
        "E": {"physics_lstm E8c": (0.575, 0.759), "Transformer E7a": (0.599, 0.736)},
    }
    for row in rows_a + rows_b:
        sc = row["name"].split("_")[-1]
        print(f"{row['variant']:<20} {sc:<10} {row['r2']:>8.4f} {row['auroc']:>8.4f}")
    print("\nBaselines:")
    for sc, models in baselines.items():
        for name, (r2, auroc) in models.items():
            print(f"  {name:<20} {sc:<10} {r2:>8.4f} {auroc:>8.4f}")

    print("\nE10 completed. Results in experiments/results/e10_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
