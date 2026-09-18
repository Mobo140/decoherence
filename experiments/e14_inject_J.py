"""Experiment E14 — Inject coupling J as a continuous physics scalar.

Goal
----
Test whether telling physics_lstm the known Hamiltonian coupling J
raises TFIM R² vs the E8c-blind model. E7c showed J/h dominates
predictability (0.88 at J=0.2, ~0.45 near the QPT).

Claim
-----
C2/C5: J is a calibrated device constant, not γ(t). Conditioning on J
        should lift mixture TFIM R² toward the J-conditional E7c numbers.

Scenarios
---------
E14a — E only (TFIM), τ=0, with-J vs no-J.
E14b — D+E cross-system, τ=0.5, with-J; compare to E8c D=0.740 E=0.575.

Metrics
-------
R² / MAE / AUROC overall and in J bins: <0.5, [0.5, 0.8), ≥0.8.

Output
------
experiments/results/e14_inject_J.csv

Usage
-----
    python -m experiments.e14_inject_J
    python -m experiments.e14_inject_J --fast
    python -m experiments.e14_inject_J --e14a
    python -m experiments.e14_inject_J --e14b
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"
WINDOW = 30
MIN_GAP = 15
J_BINS = (
    ("J_lt_0.5", 0.0, 0.5),
    ("J_0.5_0.8", 0.5, 0.8),
    ("J_ge_0.8", 0.8, 1.01),
)


def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def _lstm(*, use_J: bool, tau: float, interaction_code: float) -> LSTMPredictor:
    return LSTMPredictor(
        hidden_size=128, num_layers=2, dropout=0.3,
        use_physics_prior=True,
        adaptive_prior_r2_threshold=tau,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=interaction_code,
        use_J_scalar=use_J,
        early_stopping_patience=30,
    )


def _r2(pred: np.ndarray, actual: np.ndarray) -> float:
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _force_all_test(dataset):
    import copy
    ds = copy.copy(dataset)
    ds.test_idx = np.arange(len(dataset.sequences))
    ds.train_idx = np.array([], dtype=np.int64)
    ds.val_idx = np.array([], dtype=np.int64)
    return ds


def _predict_all(predictor, ds, horizon: float):
    idx = ds.test_idx
    if len(getattr(ds, "censored", [])) == len(ds.sequences) and len(idx) > 0:
        uncens = ~ds.censored[idx]
        if uncens.any():
            idx = idx[uncens]
    preds, actuals, js = [], [], []
    for i in idx:
        if len(ds.J_values) == len(ds.sequences):
            predictor.inference_J = float(ds.J_values[i])
        out = predictor.predict(ds.sequences[i], float(ds.t_obs[i]), horizon)
        preds.append(out.t_decoh_predicted)
        actuals.append(float(ds.t_decoh_abs[i]))
        js.append(float(ds.J_values[i]) if len(ds.J_values) == len(ds.sequences) else 0.0)
    return np.asarray(preds), np.asarray(actuals), np.asarray(js)


def _bin_rows(arm: str, scenario: str, use_J: bool, preds, actuals, js) -> List[dict]:
    rows = []
    for name, lo, hi in J_BINS:
        mask = (js >= lo) & (js < hi)
        n = int(mask.sum())
        rows.append({
            "arm": arm,
            "scenario": scenario,
            "use_J": use_J,
            "j_bin": name,
            "r2": _r2(preds[mask], actuals[mask]) if n >= 3 else float("nan"),
            "mae": float(np.mean(np.abs(preds[mask] - actuals[mask]))) if n else float("nan"),
            "n_samples": n,
            "auroc": float("nan"),
        })
    return rows


def _train(simulator, configs, *, use_J: bool, tau: float, n_epochs: int,
           verbose: bool, interaction_code: float) -> LSTMPredictor:
    ds = GenerateDatasetUseCase(simulator, store=trajectory_store()).execute(GenerateDatasetCommand(
        configs=configs,
        window_length=WINDOW,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        train_ratio=0.9,
        val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))
    pred = _lstm(use_J=use_J, tau=tau, interaction_code=interaction_code)
    TrainModelUseCase(pred).execute(TrainModelCommand(
        dataset=ds,
        n_epochs=n_epochs,
        batch_size=64,
        learning_rate=5e-4,
        regression_loss="huber",
        verbose=verbose,
        seed=42,
    ))
    return pred


def _eval_group(simulator, pred, configs, *, arm: str, scenario: str,
                use_J: bool, seed: int) -> List[dict]:
    ds = GenerateDatasetUseCase(simulator, store=trajectory_store()).execute(GenerateDatasetCommand(
        configs=configs,
        window_length=WINDOW,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=seed,
        train_ratio=0.0,
        val_ratio=0.0,
        min_window_gap=MIN_GAP,
    ))
    ds = _force_all_test(ds)
    metrics = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
    preds, actuals, js = _predict_all(pred, ds, 1.0)
    print(f"  {arm} {scenario} use_J={use_J} → {metrics.summary()}")
    overall = {
        "arm": arm,
        "scenario": scenario,
        "use_J": use_J,
        "j_bin": "all",
        "r2": metrics.r2,
        "mae": metrics.mae,
        "n_samples": metrics.n_samples,
        "auroc": metrics.risk_auroc,
    }
    return [overall] + _bin_rows(arm, scenario, use_J, preds, actuals, js)


def run_e14a(simulator, groups, n_epochs: int, verbose: bool) -> List[dict]:
    rows: List[dict] = []
    for use_J in (False, True):
        label = "e14a_J" if use_J else "e14a_noJ"
        print(f"\n=== {label} — TFIM-only ===")
        pred = _train(
            simulator, groups["E"], use_J=use_J, tau=0.0,
            n_epochs=n_epochs, verbose=verbose, interaction_code=3.0,
        )
        rows.extend(_eval_group(
            simulator, pred, groups["E"],
            arm=label, scenario="E", use_J=use_J, seed=43,
        ))
    return rows


def run_e14b(simulator, groups, n_epochs: int, verbose: bool) -> List[dict]:
    rows: List[dict] = []
    print("\n=== e14b_J — cross-system D+E ===")
    pred = _train(
        simulator, groups["D"] + groups["E"], use_J=True, tau=0.5,
        n_epochs=n_epochs, verbose=verbose, interaction_code=2.5,
    )
    for scenario, code, cfgs in (("D", 2.0, groups["D"]), ("E", 3.0, groups["E"])):
        pred.inference_interaction_code = code
        rows.extend(_eval_group(
            simulator, pred, cfgs,
            arm="e14b_J", scenario=scenario, use_J=True, seed=43,
        ))
    return rows


def main(fast: bool = False, only: Optional[str] = None) -> None:
    n_per = 40 if fast else 500
    epochs = 12 if fast else 150
    groups = build_configs(n_per_scenario=n_per, seed=42)
    simulator = QuTipSimulator()

    print(f"E14 inject J | n_per={n_per} | epochs={epochs} | window={WINDOW}")
    rows: List[dict] = []
    if only in (None, "e14a"):
        rows.extend(run_e14a(simulator, groups, epochs, verbose=True))
    if only in (None, "e14b"):
        rows.extend(run_e14b(simulator, groups, epochs, verbose=True))

    fields = ["arm", "scenario", "use_J", "j_bin", "r2", "mae", "n_samples", "auroc"]
    _save_csv(rows, RESULTS_DIR / "e14_inject_J.csv", fields)

    print(f"\n{'arm':<12} {'sc':<3} {'bin':<12} {'R²':>8} {'N':>6}")
    print("-" * 46)
    for r in rows:
        r2 = r["r2"]
        r2s = f"{r2:8.4f}" if r2 == r2 else f"{'nan':>8}"
        print(f"{r['arm']:<12} {r['scenario']:<3} {r['j_bin']:<12} {r2s} {r['n_samples']:>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--e14a", action="store_true")
    parser.add_argument("--e14b", action="store_true")
    args = parser.parse_args()
    only = "e14a" if args.e14a and not args.e14b else "e14b" if args.e14b and not args.e14a else None
    main(fast=args.fast, only=only)
