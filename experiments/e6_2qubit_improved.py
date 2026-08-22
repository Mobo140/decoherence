"""Experiment E6 — Improved two-qubit decoherence prediction.

Goal
----
Fix the two causes of poor 2-qubit scores in the first Paper 2 run:
too few trajectories, and an OLS prior that misleads on TFIM oscillations.

Claim
-----
C2/C5 on D/E: 500 trajs/scenario + adaptive prior (τ=0.7) make TFIM
learnable. Prior is gated off when OLS R² of log(C) < 0.7.

Scenarios
---------
D (2q XXZ), E (2q TFIM). Ablation + noise sweep + cross-system.

Metrics
-------
R², MAE, AUROC. Architecture: hidden 64→128, layers 1→2.

Output
------
experiments/results/e6_ablation_improved.csv
experiments/results/e6_noise_sweep_improved.csv
experiments/results/e6_cross_system_improved.csv

Usage
-----
    python -m experiments.e6_2qubit_improved
    python -m experiments.e6_2qubit_improved --fast
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.application.run_noise_sweep import NoiseSweepCommand, NoiseSweepUseCase
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path = path.with_name(f"{path.stem}_rerun{path.suffix}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def _ablation(simulator, groups, scenarios, n_epochs, suffix, verbose=True):
    """Run E6a: ablation comparing four variants on given 2-qubit scenarios."""
    results_dir = Path(__file__).parent / "results"
    all_rows = []

    for scenario_label in scenarios:
        print(f"\n{'='*60}\nScenario {scenario_label}\n{'='*60}")

        cmd = AblationCommand(
            configs=groups[scenario_label],
            variants=[
                PredictorVariant.PHYSICS_ONLY,
                PredictorVariant.LSTM_ONLY,
                PredictorVariant.PHYSICS_LSTM,
                PredictorVariant.TRANSFORMER,
            ],
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=n_epochs,
            batch_size=64,
            lr=5e-4,          # lower LR for larger model
            regression_loss="huber",
            seed=42,
            verbose=verbose,
        )

        # Use improved predictor kwargs: larger hidden, adaptive prior.
        # We monkey-patch _build_predictor via subclass to inject these.
        class ImprovedAblation(AblationStudyUseCase):
            def _build_predictor(self, variant, command):
                from src.infrastructure.ml.lstm_predictor import LSTMPredictor
                from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
                from src.infrastructure.ml.transformer_predictor import TransformerPredictor
                if variant == PredictorVariant.PHYSICS_ONLY:
                    return PhysicsOnlyPredictor(t_max=max(c.t_max for c in command.configs))
                if variant == PredictorVariant.LSTM_ONLY:
                    return LSTMPredictor(
                        hidden_size=128, num_layers=2, dropout=0.3,
                        use_physics_prior=False, adaptive_prior_r2_threshold=0.0,
                    )
                if variant == PredictorVariant.PHYSICS_LSTM:
                    return LSTMPredictor(
                        hidden_size=128, num_layers=2, dropout=0.3,
                        use_physics_prior=True, adaptive_prior_r2_threshold=0.7,
                    )
                if variant == PredictorVariant.TRANSFORMER:
                    return TransformerPredictor()
                raise ValueError(variant)

        results = ImprovedAblation(simulator, store=trajectory_store()).execute(cmd)
        for r in results:
            row = r.as_dict()
            row["scenario"] = scenario_label
            all_rows.append(row)

    fname = f"e6_ablation{suffix}.csv"
    fieldnames = ["scenario", "name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    _save_csv(all_rows, results_dir / fname, fieldnames)

    print(f"\n{'Scenario':<10} {'Variant':<16} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 54)
    for r in all_rows:
        print(f"{r['scenario']:<10} {r['variant']:<16} {r['r2']:>8.4f} {r['mae']:>8.4f} {r['auroc']:>8.4f}")

    return all_rows


def _noise_sweep(simulator, groups, scenarios, n_epochs, suffix, verbose=True):
    """Run E6b: noise robustness with improved model on given 2-qubit scenarios."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor
    results_dir = Path(__file__).parent / "results"
    configs = [cfg for s in scenarios for cfg in groups[s]]

    cmd = NoiseSweepCommand(
        configs=configs,
        sigmas=[0.0, 0.01, 0.02, 0.05, 0.10],
        window_length=50,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=42,
        verbose=verbose,
    )

    class ImprovedNoiseSweep(NoiseSweepUseCase):
        def _build_predictor(self):
            return LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=True, adaptive_prior_r2_threshold=0.7,
            )

    results = ImprovedNoiseSweep(simulator).execute(cmd)
    rows = [r.as_dict() for r in results]
    fname = f"e6_noise_sweep{suffix}.csv"
    fieldnames = ["name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    _save_csv(rows, results_dir / fname, fieldnames)

    print(f"\n{'σ_noise':>9} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 38)
    for r in results:
        flag = " ✓" if r.risk_auroc >= 0.90 else " ✗"
        print(f"{r.spec.noise.sigma:>9.3f} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f}{flag}")

    return rows


def _cross_system(simulator, groups, train_scenarios, eval_scenarios, n_epochs, suffix, verbose=True):
    """Run E6c: cross-system generalisation with improved model."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor
    results_dir = Path(__file__).parent / "results"

    train_configs = [cfg for s in train_scenarios for cfg in groups[s]]
    eval_groups = {s: groups[s] for s in eval_scenarios}

    cmd = CrossSystemCommand(
        train_configs=train_configs,
        eval_groups=eval_groups,
        window_length=50,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=42,
        verbose=verbose,
    )

    class ImprovedCrossSystem(CrossSystemUseCase):
        def _build_predictor(self):
            return LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=True, adaptive_prior_r2_threshold=0.7,
            )

    results = ImprovedCrossSystem(simulator, store=trajectory_store()).execute(cmd)
    rows = [r.as_dict() for r in results]
    fname = f"e6_cross_system{suffix}.csv"
    fieldnames = ["name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    _save_csv(rows, results_dir / fname, fieldnames)

    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for r in results:
        print(f"{r.spec.name:>20} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f} {r.n_samples:>6}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_per  = 50  if fast else 500
    epochs = 10  if fast else 150

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    suffix = "_improved"
    scenarios = ["D", "E"]

    print(f"\n{'='*70}")
    print(f"E6a — Ablation (D, E) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    _ablation(simulator, groups, scenarios, epochs, suffix)

    print(f"\n{'='*70}")
    print(f"E6b — Noise sweep (D+E) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    _noise_sweep(simulator, groups, scenarios, epochs, suffix)

    print(f"\n{'='*70}")
    print(f"E6c — Cross-system (train D+E, eval D/E) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    _cross_system(simulator, groups, ["D", "E"], ["D", "E"], epochs, suffix)

    print("\nE6 completed. Results in experiments/results/e6_*_improved.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
