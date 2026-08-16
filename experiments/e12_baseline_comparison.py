"""Experiment E12 — Classical baseline comparison.

Goal
----
Check that physics_lstm beats a non-ML stretched-exponential curve_fit
baseline on time-dependent γ (claim C2).  Without this, C2 is not
falsifiable: a reviewer can ask whether ordinary curve_fit already
explains the R².

Claim
-----
C2: physics+LSTM residual outperforms physics-only AND stretched-exp
    for time-dependent γ(t).

Scenarios
---------
B (1q σ₋ time-dep), C (1q σ_z time-dep), D (2q XXZ), E (2q TFIM).

Metrics
-------
R², MAE. physics_only = OLS log C; stretched_exp = curve_fit C0·exp(-(t/τ)^β).

Output
------
experiments/results/e12_baseline_comparison.csv

Usage
-----
    python -m experiments.e12_baseline_comparison
    python -m experiments.e12_baseline_comparison --fast
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"

VARIANTS = [
    PredictorVariant.PHYSICS_ONLY,
    PredictorVariant.STRETCHED_EXP,
    PredictorVariant.PHYSICS_LSTM,
]


def main(fast: bool = False) -> None:
    results_dir = RESULTS_DIR
    results_dir.mkdir(exist_ok=True)

    n_per = 20 if fast else 100
    n_epochs = 15 if fast else 80
    scenarios = ["B", "C", "D", "E"]

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    all_rows = []

    for scenario_label in scenarios:
        print(f"\n{'='*60}")
        print(f"E12 — Scenario {scenario_label}  n_per={n_per}  epochs={n_epochs}")
        print(f"{'='*60}")

        cmd = AblationCommand(
            configs=groups[scenario_label],
            variants=VARIANTS,
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=n_epochs,
            batch_size=32,
            lr=1e-3,
            regression_loss="huber",
            seed=42,
            verbose=True,
        )

        results = AblationStudyUseCase(simulator, store=trajectory_store()).execute(cmd)

        for r in results:
            row = r.as_dict()
            row["scenario"] = scenario_label
            all_rows.append(row)

    out_path = results_dir / "e12_baseline_comparison.csv"
    fieldnames = [
        "scenario", "name", "variant", "window_fraction", "noise_sigma",
        "mae", "rmse", "r2", "mape", "auroc", "n_samples",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {out_path}")
    print(f"\n{'Scenario':<10} {'Variant':<16} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 54)
    for r in all_rows:
        print(
            f"{r['scenario']:<10} {r['variant']:<16} "
            f"{r['r2']:>8.4f} {r['mae']:>8.4f} {r['auroc']:>8.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
