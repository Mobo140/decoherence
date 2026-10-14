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
experiments/results/e7a_transformer_cross[_<tag>][_fast].csv
experiments/results/e7b_tau_sweep[_<tag>][_fast].csv
experiments/results/e7c_J_sweep[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e7_paper2_extended
    python -m experiments.e7_paper2_extended --fast
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np

from experiments import common
from src.application import backtest
from src.application import generate_dataset
from src.application import run_cross_system
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.domain import value_objects
from src.models import registry
from src.simulation import simulator as simulator_lib

E7A_COLUMNS = (
    "experiment", "name", "variant", "window_fraction", "noise_sigma",
    "mae", "rmse", "r2", "mape", "auroc", "n_samples")
E7B_COLUMNS = ("scenario", "tau", "r2", "mae", "rmse", "auroc", "n_samples")
E7C_COLUMNS = ("J", "J_over_h", "r2", "mae", "rmse", "auroc", "n_samples")
TAUS = (0.0, 0.3, 0.5, 0.7, 0.9)
J_VALUES = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
# Seed of the configuration generator of the J sweep.
J_CONFIG_SEED = 123


def tau_predictor(tau: float) -> ports.IPredictor:
    """Builds the physics + LSTM predictor of E7b for one threshold.

    R2 replicates E7b with this factory so that both use one model.

    Args:
        tau: Adaptive-prior threshold on the OLS R² of log C.

    Returns:
        The untrained predictor.
    """
    return registry.build_model(
        "physics_lstm", hidden_size=128, num_layers=2, dropout=0.3,
        adaptive_prior_r2_threshold=tau)


def _train_and_score(predictor, dataset, n_epochs: int, seed: int,
                     verbose: bool):
    """Trains with the E7 settings and backtests on the same dataset."""
    train_model.TrainModelUseCase(predictor).execute(
        train_model.TrainModelCommand(
            dataset=dataset,
            n_epochs=n_epochs,
            batch_size=64,
            learning_rate=5e-4,
            regression_loss="huber",
            verbose=verbose,
            seed=seed,
        ))
    return backtest.BacktestUseCase(predictor).execute(
        backtest.BacktestCommand(dataset=dataset, horizon=1.0))


def run_e7a(simulator, groups, n_epochs: int,
            verbose: bool = True) -> list[dict[str, object]]:
    """E7a: large Transformer trained on D+E, scored on D and E.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per evaluation scenario, with the keys of
        `E7A_COLUMNS`.
    """
    results = run_cross_system.CrossSystemUseCase(
        simulator, store=scenario_lib.trajectory_store(),
        # d_model=128, 4 heads, 3 layers, feed-forward 512.
        predictor_factory=functools.partial(
            registry.build_model, "transformer_large"),
    ).execute(run_cross_system.CrossSystemCommand(
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
    ))
    rows = [{**r.as_dict(), "experiment": "E7a_transformer_cross"}
            for r in results]
    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for row in rows:
        print(f"{row['name']:>20} {row['r2']:>8.4f} {row['mae']:>8.4f} "
              f"{row['auroc']:>8.4f} {row['n_samples']:>6}")
    return rows


def run_e7b(simulator, groups, n_epochs: int,
            verbose: bool = True) -> list[dict[str, object]]:
    """E7b: sweeps the adaptive-prior threshold τ on D and on E.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per scenario and τ, with the keys of `E7B_COLUMNS`.
    """
    rows = []
    for scenario in ("D", "E"):
        print(f"\n{'=' * 60}\nτ sweep — Scenario {scenario}\n{'=' * 60}")
        dataset = generate_dataset.GenerateDatasetUseCase(
            simulator, store=scenario_lib.trajectory_store()
        ).execute(generate_dataset.GenerateDatasetCommand(
            configs=groups[scenario],
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=42,
        ))
        for tau in TAUS:
            print(f"\n  τ = {tau:.1f}")
            metrics = _train_and_score(
                tau_predictor(tau), dataset, n_epochs, 42, verbose)
            rows.append({
                "scenario": scenario,
                "tau": tau,
                "r2": metrics.r2,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "auroc": metrics.risk_auroc,
                "n_samples": metrics.n_samples,
            })
            print(f"  → R²={metrics.r2:.4f}  MAE={metrics.mae:.4f}  "
                  f"AUROC={metrics.risk_auroc:.4f}")

    print(f"\n{'Scenario':>8} {'τ':>5} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 42)
    for row in rows:
        print(f"{row['scenario']:>8} {row['tau']:>5.1f} {row['r2']:>8.4f} "
              f"{row['mae']:>8.4f} {row['auroc']:>8.4f}")
    return rows


def run_e7c(simulator, n_per_j: int, n_epochs: int,
            verbose: bool = True) -> list[dict[str, object]]:
    """E7c: sweeps the TFIM coupling J and trains a Transformer per J.

    The phase transition is at J = h = 1.0 (h is fixed in
    `TwoQubitSystem`).

    Args:
        simulator: The simulator.
        n_per_j: Trajectories per J.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per J with test samples, with the keys of `E7C_COLUMNS`.
    """
    rng = np.random.default_rng(J_CONFIG_SEED)
    generator = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store())
    rows = []
    for coupling in J_VALUES:
        print(f"\n{'=' * 60}\nTFIM J={coupling:.1f}\n{'=' * 60}")
        configs = scenario_lib.fixed_coupling_configs(
            n_per_j, coupling, rng,
            value_objects.DecoherenceCriterion.COHERENCE,
            value_objects.InteractionType.TFIM)
        dataset = generator.execute(generate_dataset.GenerateDatasetCommand(
            configs=configs,
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=42,
        ))
        if len(dataset.test_idx) == 0:
            print(f"  No test samples — skipping J={coupling}")
            continue
        metrics = _train_and_score(
            registry.build_model("transformer_large"), dataset, n_epochs,
            42, verbose)
        rows.append({
            "J": coupling,
            "J_over_h": coupling,     # h = 1.0 fixed
            "r2": metrics.r2,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "auroc": metrics.risk_auroc,
            "n_samples": metrics.n_samples,
        })
        print(f"  → R²={metrics.r2:.4f}  MAE={metrics.mae:.4f}  "
              f"AUROC={metrics.risk_auroc:.4f}  N={metrics.n_samples}")

    print(f"\n{'J':>6} {'J/h':>6} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 48)
    for row in rows:
        marker = " ← phase transition" if abs(row["J"] - 1.0) < 0.05 else ""
        print(f"{row['J']:>6.1f} {row['J_over_h']:>6.1f} "
              f"{row['r2']:>8.4f} {row['mae']:>8.4f} {row['auroc']:>8.4f} "
              f"{row['n_samples']:>6}{marker}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E7a, E7b and E7c and writes their CSVs."""
    args = common.parse_args(__doc__, argv=argv, seeds=None, out=False)
    n_per, n_per_j, epochs = (50, 30, 10) if args.fast else (500, 200, 150)
    config = {"n_per": n_per, "n_per_J": n_per_j, "n_epochs": epochs}
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'=' * 70}\nE7a — Transformer cross-system (D+E, "
          f"n_per={n_per}, epochs={epochs})\n{'=' * 70}")
    rows = run_e7a(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e7a_transformer_cross", args),
        columns=E7A_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE7b — τ sweep for physics_lstm (D and E, "
          f"n_per={n_per}, epochs={epochs})\n{'=' * 70}")
    rows = run_e7b(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e7b_tau_sweep", args),
        columns=E7B_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE7c — TFIM J sweep (n_per_J={n_per_j}, "
          f"epochs={epochs})\n{'=' * 70}")
    rows = run_e7c(simulator, n_per_j, epochs)
    common.write_results(
        rows, common.results_path("e7c_J_sweep", args),
        columns=E7C_COLUMNS, seeds=[42], config=config)
    print("\nE7 completed. Results in experiments/results/e7*.csv")


if __name__ == "__main__":
    main()
