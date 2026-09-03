"""Experiment E10 (TFIM-specialized) — Larger window / hidden for scenario E.

Goal
----
Raise TFIM R² after E9 context codes missed 0.70. Hypotheses: TFIM-only
model, window 40–50, hidden=256, τ=0 (OLS prior hurts TFIM oscillations).

Claim
-----
C2: physics+LSTM is needed when the classical prior fails (TFIM).
C5: one mixed model (E10c) vs specialized TFIM-only (E10a/b/d).

Scenarios
---------
E (TFIM) for a/b/d; D+E for c. Codes: D=2, E=3.
Does not write e10a_cross_system.csv (that file belongs to e10_fixed_context).

Metrics
-------
R² on uncensored test windows. Target E ≥ 0.70. Best here: E10c E=0.472.

Output
------
experiments/results/e10_tfim_c.csv   # mixed D+E; a/b/d TFIM-only were R²<0, not kept

Usage
-----
    python -m experiments.e10_tfim_specialized
    python -m experiments.e10_tfim_specialized --fast
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
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.domain.value_objects import DissipatorType, InteractionType, PredictorVariant
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.persistence.stores import HamiltonianModelRegistry
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"
_FAST = False


def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def _make_tfim_lstm(hidden: int, tau: float, int_code: float) -> LSTMPredictor:
    return LSTMPredictor(
        hidden_size=hidden,
        num_layers=2,
        dropout=0.3,
        use_physics_prior=True,
        adaptive_prior_r2_threshold=tau,
        risk_pos_weight=0.0,
        augment_sigma=0.02,
        inference_interaction_code=int_code,
        inference_dissipator_code=0.0,
        early_stopping_patience=30,
    )


def _train_single(
    simulator,
    configs,
    *,
    window: int,
    hidden: int,
    tau: float,
    int_code: float,
    n_epochs: int,
    min_gap: int,
    verbose: bool,
):
    ds = GenerateDatasetUseCase(simulator, store=trajectory_store()).execute(GenerateDatasetCommand(
        configs=configs,
        window_length=window,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        min_window_gap=min_gap,
    ))
    pred = _make_tfim_lstm(hidden, tau, int_code)
    TrainModelUseCase(pred).execute(TrainModelCommand(
        dataset=ds,
        n_epochs=n_epochs,
        batch_size=64,
        learning_rate=5e-4,
        regression_loss="huber",
        verbose=verbose,
    ))
    metrics = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
    return metrics, pred


def run_e10a(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    window = 20 if _FAST else 40
    hidden = 64 if _FAST else 256
    print(f"\n{'='*60}\nE10a — TFIM-only  window={window}  hidden={hidden}  τ=0\n{'='*60}")
    metrics, pred = _train_single(
        simulator, groups["E"],
        window=window, hidden=hidden, tau=0.0, int_code=3.0,
        n_epochs=n_epochs, min_gap=max(window // 2, 1), verbose=verbose,
    )
    updated = HamiltonianModelRegistry().maybe_update(
        pred, metrics, 2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS,
    )
    print(f"  → {metrics.summary()}")
    print(f"  registry best_TFIM.pt updated={updated}")
    row = {
        "name": "e10a_tfim_only",
        "variant": PredictorVariant.PHYSICS_LSTM.value,
        "scenario": "E",
        "mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2,
        "mape": metrics.mape, "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
    }
    _save_csv([row], RESULTS_DIR / "e10_tfim_a.csv",
              ["name", "variant", "scenario", "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return [row]


def run_e10b(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    window = 25 if _FAST else 50
    hidden = 64 if _FAST else 256
    print(f"\n{'='*60}\nE10b — TFIM-only  window={window}  hidden={hidden}  τ=0  code=3\n{'='*60}")
    metrics, _ = _train_single(
        simulator, groups["E"],
        window=window, hidden=hidden, tau=0.0, int_code=3.0,
        n_epochs=n_epochs, min_gap=max(window // 2, 1), verbose=verbose,
    )
    print(f"  → {metrics.summary()}")
    row = {
        "name": "e10b_tfim_window50",
        "variant": PredictorVariant.PHYSICS_LSTM.value,
        "scenario": "E",
        "mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2,
        "mape": metrics.mape, "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
    }
    _save_csv([row], RESULTS_DIR / "e10_tfim_b.csv",
              ["name", "variant", "scenario", "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return [row]


def run_e10c(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    window = 20 if _FAST else 40
    hidden = 64 if _FAST else 256
    print(f"\n{'='*60}\nE10c — cross-system D+E  window={window}  hidden={hidden}\n{'='*60}")

    class E10Cross(CrossSystemUseCase):
        def _build_predictor(self):
            return _make_tfim_lstm(hidden=hidden, tau=0.0, int_code=2.5)

    cmd = CrossSystemCommand(
        train_configs=groups["D"] + groups["E"],
        eval_groups={"D": groups["D"], "E": groups["E"]},
        window_length=window,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=42,
        verbose=verbose,
        min_window_gap=max(window // 2, 1),
        eval_interaction_codes={"D": 2.0, "E": 3.0},
    )
    results = E10Cross(simulator).execute(cmd)
    rows = []
    for r in results:
        row = r.as_dict()
        row["scenario"] = r.spec.name.replace("cross_system_", "")
        rows.append(row)
    _save_csv(rows, RESULTS_DIR / "e10_tfim_c.csv",
              ["name", "variant", "scenario", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    print(f"\n{'Scenario':>8} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    for r in results:
        print(f"{r.spec.name:>8} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f}")
    return rows


def run_e10d(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """TFIM-only with E8 window (30), so the test split stays non-empty."""
    window = 20 if _FAST else 30
    hidden = 64 if _FAST else 256
    print(f"\n{'='*60}\nE10d — TFIM-only  window={window}  hidden={hidden}  τ=0\n{'='*60}")
    metrics, pred = _train_single(
        simulator, groups["E"],
        window=window, hidden=hidden, tau=0.0, int_code=3.0,
        n_epochs=n_epochs, min_gap=max(window // 2, 1), verbose=verbose,
    )
    updated = HamiltonianModelRegistry().maybe_update(
        pred, metrics, 2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS,
    )
    print(f"  → {metrics.summary()}")
    print(f"  registry best_TFIM.pt updated={updated}")
    row = {
        "name": "e10d_tfim_window30",
        "variant": PredictorVariant.PHYSICS_LSTM.value,
        "scenario": "E",
        "mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2,
        "mape": metrics.mape, "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
    }
    _save_csv([row], RESULTS_DIR / "e10_tfim_d.csv",
              ["name", "variant", "scenario", "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return [row]


def main(fast: bool = False) -> None:
    n_per = 40 if fast else 300
    epochs = 15 if fast else 120
    global _FAST
    _FAST = fast
    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    print(f"E10 TFIM specialized | n_per={n_per} | epochs={epochs}")
    run_e10a(simulator, groups, n_epochs=epochs)
    run_e10b(simulator, groups, n_epochs=epochs)
    run_e10c(simulator, groups, n_epochs=epochs)
    print("\nE10 completed. Results in experiments/results/e10_tfim_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
