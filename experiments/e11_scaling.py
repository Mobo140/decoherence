"""Experiment E11 — Scaling and window optimisation.

Goal
----
Test whether E8c TFIM R²=0.575 is limited by window length (too few
pre-T₂ windows) or by training-set size.

Claim
-----
C5: window=20 + E8c regularisation (E11a) or 1000 trajs (E11b) raise
    cross-system R² vs E8c / E7a.

Scenarios
---------
D (2q XXZ), E (2q TFIM), train D+E.

Metrics
-------
E11a Transformer: target D R²≥0.75, E R²≥0.60.
E11b physics_lstm, 1000 trajs: target E R²≥0.62.

Output
------
experiments/results/e11a_cross_system.csv
experiments/results/e11b_cross_system.csv

Usage
-----
    python -m experiments.e11_scaling
    python -m experiments.e11_scaling --fast
    python -m experiments.e11_scaling --e11a
    python -m experiments.e11_scaling --e11b
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"


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


# 4-class system-type codes matching generate_dataset.py
_INFERENCE_CODES = {"A": 0.0, "B": 0.0, "C": 1.0, "D": 2.0, "E": 3.0}


def _eval_per_scenario(predictor, simulator, groups, window, min_gap,
                       scenarios=("D", "E"), set_inference_codes: bool = False,
                       eval_seed: int = 43):
    """Evaluate predictor on each scenario's test split.

    set_inference_codes: if True, update predictor.inference_interaction_code
    before each eval (required for physics_lstm with context codes).
    """
    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())
    rows = []
    for sc in scenarios:
        if set_inference_codes and hasattr(predictor, "inference_interaction_code"):
            predictor.inference_interaction_code = _INFERENCE_CODES.get(sc, 0.0)

        eval_ds = gen_uc.execute(GenerateDatasetCommand(
            configs=groups[sc],
            window_length=window,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=eval_seed,
            train_ratio=0.0, val_ratio=0.0,
            min_window_gap=min_gap,
        ))
        eval_ds = _force_all_test(eval_ds)
        m = BacktestUseCase(predictor).execute(BacktestCommand(dataset=eval_ds, horizon=1.0))
        print(f"  [{sc}] → {m.summary()}")
        rows.append(dict(
            scenario=sc, mae=m.mae, rmse=m.rmse, r2=m.r2,
            mape=m.mape, auroc=m.risk_auroc, n_samples=m.n_samples,
        ))
    return rows


# ---------------------------------------------------------------------------
# E11a — Transformer, window=20, E8c training settings
# ---------------------------------------------------------------------------

def run_e11a(simulator, groups, n_epochs: int, n_per: int, verbose: bool = True,
             *, seed: int = 42, window: int = 20, out_path=None,
             eval_groups=None) -> list:
    """Defaults reproduce the published E11a run; the keyword-only arguments
    exist so R3 can replicate it across seeds and window lengths without
    duplicating the configuration (see experiments/r3_multiseed_scaling.py)."""
    from src.infrastructure.ml.transformer_predictor import TransformerPredictor

    WINDOW  = window
    MIN_GAP = 10
    print(f"=== E11a: Transformer cross-system | window={WINDOW} | n_per={n_per} ===")

    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())
    train_ds = gen_uc.execute(GenerateDatasetCommand(
        configs=groups["D"] + groups["E"],
        window_length=WINDOW, horizon=1.0,
        samples_per_trajectory=5, seed=seed,
        train_ratio=0.9, val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))
    print(f"  Train samples: {len(train_ds.train_idx)}  Val: {len(train_ds.val_idx)}")

    predictor = TransformerPredictor(
        d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1,
    )
    TrainModelUseCase(predictor).execute(TrainModelCommand(
        dataset=train_ds, n_epochs=n_epochs,
        batch_size=64, learning_rate=5e-4,
        regression_loss="huber", verbose=verbose,
        seed=seed,
    ))

    # eval_groups defaults to the training configurations: the published run
    # scores the model on the same physical systems it trained on, with fresh
    # trajectories. R4 passes disjoint configurations to measure what that is
    # worth (see experiments/r4_config_overlap.py).
    rows = _eval_per_scenario(predictor, simulator,
                              groups if eval_groups is None else eval_groups,
                              WINDOW, MIN_GAP, eval_seed=seed + 1)
    for r in rows:
        r["variant"] = f"transformer_w{WINDOW}"
        r["window"] = WINDOW

    _save_csv(rows, out_path or RESULTS_DIR / "e11a_cross_system.csv",
              ["scenario", "variant", "window", "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return rows


# ---------------------------------------------------------------------------
# E11b — physics_lstm, 1000 trajs, window=20
# ---------------------------------------------------------------------------

def run_e11b(simulator, groups_1000, n_epochs: int, verbose: bool = True,
             *, seed: int = 42, out_path=None) -> list:
    """Defaults reproduce the published E11b run; see run_e11a for why the
    keyword-only arguments exist. Training-set size is chosen by the caller
    through ``groups_1000``, which is what R3 varies."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    WINDOW  = 20
    MIN_GAP = 15
    print(f"=== E11b: physics_lstm cross-system | 1000 trajs | window={WINDOW} ===")

    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())
    train_ds = gen_uc.execute(GenerateDatasetCommand(
        configs=groups_1000["D"] + groups_1000["E"],
        window_length=WINDOW, horizon=1.0,
        samples_per_trajectory=5, seed=seed,
        train_ratio=0.9, val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))
    print(f"  Train samples: {len(train_ds.train_idx)}  Val: {len(train_ds.val_idx)}")

    predictor = LSTMPredictor(
        hidden_size=128, num_layers=2, dropout=0.3,
        use_physics_prior=True, adaptive_prior_r2_threshold=0.5,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=2.0,   # overridden per scenario at eval
        inference_dissipator_code=0.0,
        early_stopping_patience=50,       # 3x dataset → needs more patience
    )
    TrainModelUseCase(predictor).execute(TrainModelCommand(
        dataset=train_ds, n_epochs=n_epochs,
        batch_size=64, learning_rate=5e-4,
        regression_loss="huber", verbose=verbose,
        seed=seed,
    ))

    rows = _eval_per_scenario(predictor, simulator, groups_1000, WINDOW, MIN_GAP,
                              set_inference_codes=True, eval_seed=seed + 1)
    for r in rows:
        r["variant"] = "physics_lstm_1000"
        r["window"] = WINDOW

    # Use groups_1000 for eval too (1000 trajs provides larger test set)
    _save_csv(rows, out_path or RESULTS_DIR / "e11b_cross_system.csv",
              ["scenario", "variant", "window", "mae", "rmse", "r2", "mape", "auroc", "n_samples"])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False, only_a: bool = False, only_b: bool = False) -> None:
    n_per_500  = 50  if fast else 500
    n_per_1000 = 100 if fast else 1000
    epochs     = 10  if fast else 150

    simulator  = QuTipSimulator()
    groups_500 = build_configs(n_per_scenario=n_per_500,  seed=42)

    rows_a, rows_b = [], []

    if not only_b:
        print(f"\n{'='*70}")
        print(f"E11a — Transformer window=20+E8c | n_per={n_per_500} | epochs={epochs}")
        print(f"{'='*70}")
        rows_a = run_e11a(simulator, groups_500, n_epochs=epochs, n_per=n_per_500)

    if not only_a:
        print(f"\n{'='*70}")
        print(f"E11b — physics_lstm 1000 trajs   | n_per={n_per_1000} | epochs={epochs}")
        print(f"{'='*70}")
        groups_1000 = build_configs(n_per_scenario=n_per_1000, seed=42)
        rows_b = run_e11b(simulator, groups_1000, n_epochs=epochs)

    print("\n\n=== SUMMARY vs baselines ===")
    print(f"\n{'Model':<28} {'Scen':>5} {'R²':>8} {'AUROC':>8}")
    print("-" * 55)
    baselines = [
        ("E8c physics_lstm",         "D", 0.740, 0.789),
        ("E8c physics_lstm",         "E", 0.575, 0.759),
        ("E10b Transformer w=30",    "D", 0.734, 0.890),
        ("E10b Transformer w=30",    "E", 0.567, 0.801),
    ]
    for name, sc, r2, auroc in baselines:
        print(f"  {name:<26} {sc:>5} {r2:>8.4f} {auroc:>8.4f}")
    print()
    for r in rows_a + rows_b:
        print(f"  {r['variant']:<26} {r['scenario']:>5} {r['r2']:>8.4f} {r['auroc']:>8.4f}")

    print("\nE11 completed. Results in experiments/results/e11_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",  action="store_true")
    parser.add_argument("--e11a",  action="store_true", dest="only_a")
    parser.add_argument("--e11b",  action="store_true", dest="only_b")
    args = parser.parse_args()
    main(fast=args.fast, only_a=args.only_a, only_b=args.only_b)
