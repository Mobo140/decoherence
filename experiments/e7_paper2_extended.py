"""Experiment E7 — Extended two-qubit study (Paper 2 completion).

Goal
----
Finish Paper 2 physics: Transformer cross-system, per-Hamiltonian τ,
and TFIM predictability vs J across the phase transition.

Claim
-----
C5: the best single-scenario model (Transformer) also wins cross-system.
C2: optimal adaptive-prior τ differs by Hamiltonian (E7b).
Predictability of TFIM varies with J at fixed h=1 (E7c; QPT at J=h=1).

Scenarios
---------
D (2q XXZ), E (2q TFIM). E7c: E only, J ∈ {0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0}.

Metrics
-------
E7a: R² / AUROC per eval scenario (500 trajs, D+E train).
E7b: R² vs τ ∈ {0.0, 0.3, 0.5, 0.7, 0.9} on D and on E.
E7c: R² vs J for Transformer.

Output
------
experiments/results/e7a_transformer_cross.csv
experiments/results/e7b_tau_sweep.csv
experiments/results/e7c_J_sweep.csv

Usage
-----
    python -m experiments.e7_paper2_extended
    python -m experiments.e7_paper2_extended --fast
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store, _bernstein_coeffs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.domain.value_objects import (
    DecoherenceCriterion, DissipatorConfig, DissipatorType,
    InteractionType, PredictorVariant, QubitCount, SystemConfig,
)
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"


def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# E7a — Transformer cross-system
# ---------------------------------------------------------------------------

def run_e7a(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """Train large Transformer on D+E, eval on D/E separately."""

    class TransformerCrossSystem(CrossSystemUseCase):
        def _build_predictor(self):
            from src.infrastructure.ml.transformer_predictor import TransformerPredictor
            # Larger Transformer: d_model=128, 4 heads, 3 layers, ff=512
            return TransformerPredictor(
                d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1,
            )

    cmd = CrossSystemCommand(
        train_configs=groups["D"] + groups["E"],
        eval_groups={"D": groups["D"], "E": groups["E"]},
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
    results = TransformerCrossSystem(simulator, store=trajectory_store()).execute(cmd)
    rows = []
    for r in results:
        row = r.as_dict()
        row["experiment"] = "E7a_transformer_cross"
        rows.append(row)

    fieldnames = ["experiment", "name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    _save_csv(rows, RESULTS_DIR / "e7a_transformer_cross.csv", fieldnames)

    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for r in results:
        print(f"{r.spec.name:>20} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f} {r.n_samples:>6}")
    return rows


# ---------------------------------------------------------------------------
# E7b — Adaptive-prior threshold (τ) sweep
# ---------------------------------------------------------------------------

def run_e7b(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """Sweep τ for physics_lstm on D and E to find optimal threshold."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    scenarios = ["D", "E"]
    rows = []

    for scenario in scenarios:
        print(f"\n{'='*60}\nτ sweep — Scenario {scenario}\n{'='*60}")

        gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())
        dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=groups[scenario],
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=42,
        ))

        for tau in thresholds:
            print(f"\n  τ = {tau:.1f}")
            predictor = LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=True,
                adaptive_prior_r2_threshold=tau,
            )
            TrainModelUseCase(predictor).execute(TrainModelCommand(
                dataset=dataset,
                n_epochs=n_epochs,
                batch_size=64,
                learning_rate=5e-4,
                regression_loss="huber",
                verbose=verbose,
            ))
            metrics = BacktestUseCase(predictor).execute(
                BacktestCommand(dataset=dataset, horizon=1.0)
            )
            rows.append({
                "scenario": scenario,
                "tau": tau,
                "r2": metrics.r2,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "auroc": metrics.risk_auroc,
                "n_samples": metrics.n_samples,
            })
            print(f"  → R²={metrics.r2:.4f}  MAE={metrics.mae:.4f}  AUROC={metrics.risk_auroc:.4f}")

    _save_csv(rows, RESULTS_DIR / "e7b_tau_sweep.csv",
              ["scenario", "tau", "r2", "mae", "rmse", "auroc", "n_samples"])

    print(f"\n{'Scenario':>8} {'τ':>5} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 42)
    for r in rows:
        print(f"{r['scenario']:>8} {r['tau']:>5.1f} {r['r2']:>8.4f} {r['mae']:>8.4f} {r['auroc']:>8.4f}")
    return rows


# ---------------------------------------------------------------------------
# E7c — TFIM J-coupling sweep
# ---------------------------------------------------------------------------

def _make_tfim_configs_fixed_J(
    n: int, J: float, rng: np.random.Generator
) -> List[SystemConfig]:
    """Generate TFIM configs with a fixed coupling constant J."""
    shapes = ["random", "peak", "step_up", "step_down", "monotone_increasing",
              "monotone_decreasing"]
    shape_cycle = itertools.cycle(shapes)
    configs = []
    for _ in range(n):
        shape = next(shape_cycle)
        coeffs = _bernstein_coeffs(shape, rng, gamma_max=0.8)
        configs.append(SystemConfig(
            n_qubits=QubitCount.TWO,
            omega=float(rng.uniform(0.5, 2.0)),
            J=J,
            dissipator=DissipatorConfig.time_dependent(coeffs, DissipatorType.SIGMA_MINUS),
            interaction_type=InteractionType.TFIM,
            t_max=20.0,
            dt=0.1,
            decoherence_criterion=DecoherenceCriterion.COHERENCE,
        ))
    return configs


def run_e7c(simulator, n_per_J: int, n_epochs: int, verbose: bool = True) -> list:
    """Sweep TFIM coupling J ∈ {0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0}.

    Phase transition at J = h_field = 1.0 (hardcoded in TwoQubitSystem).
    Train Transformer on each J-bucket and report R² and AUROC.
    """
    from src.infrastructure.ml.transformer_predictor import TransformerPredictor

    J_values = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    rng = np.random.default_rng(123)
    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())
    rows = []

    for J in J_values:
        print(f"\n{'='*60}\nTFIM J={J:.1f}  (J/h = {J:.1f})\n{'='*60}")
        configs = _make_tfim_configs_fixed_J(n_per_J, J, rng)

        dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=configs,
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=42,
        ))
        if len(dataset.test_idx) == 0:
            print(f"  No test samples — skipping J={J}")
            continue

        predictor = TransformerPredictor(
            d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1,
        )
        TrainModelUseCase(predictor).execute(TrainModelCommand(
            dataset=dataset,
            n_epochs=n_epochs,
            batch_size=64,
            learning_rate=5e-4,
            regression_loss="huber",
            verbose=verbose,
        ))
        metrics = BacktestUseCase(predictor).execute(
            BacktestCommand(dataset=dataset, horizon=1.0)
        )
        rows.append({
            "J": J,
            "J_over_h": J,     # h=1.0 fixed
            "r2": metrics.r2,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "auroc": metrics.risk_auroc,
            "n_samples": metrics.n_samples,
        })
        print(f"  → R²={metrics.r2:.4f}  MAE={metrics.mae:.4f}  "
              f"AUROC={metrics.risk_auroc:.4f}  N={metrics.n_samples}")

    _save_csv(rows, RESULTS_DIR / "e7c_J_sweep.csv",
              ["J", "J_over_h", "r2", "mae", "rmse", "auroc", "n_samples"])

    print(f"\n{'J':>6} {'J/h':>6} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 48)
    for r in rows:
        marker = " ← phase transition" if abs(r["J"] - 1.0) < 0.05 else ""
        print(f"{r['J']:>6.1f} {r['J_over_h']:>6.1f} {r['r2']:>8.4f} "
              f"{r['mae']:>8.4f} {r['auroc']:>8.4f} {r['n_samples']:>6}{marker}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_per   = 50  if fast else 500
    n_per_J = 30  if fast else 200
    epochs  = 10  if fast else 150

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'='*70}")
    print(f"E7a — Transformer cross-system (D+E, n_per={n_per}, epochs={epochs})")
    print(f"{'='*70}")
    run_e7a(simulator, groups, n_epochs=epochs)

    print(f"\n{'='*70}")
    print(f"E7b — τ sweep for physics_lstm (D and E, n_per={n_per}, epochs={epochs})")
    print(f"{'='*70}")
    run_e7b(simulator, groups, n_epochs=epochs)

    print(f"\n{'='*70}")
    print(f"E7c — TFIM J sweep (n_per_J={n_per_J}, epochs={epochs})")
    print(f"{'='*70}")
    run_e7c(simulator, n_per_J=n_per_J, n_epochs=epochs)

    print("\nE7 completed. Results in experiments/results/e7_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
