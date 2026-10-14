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
experiments/results/e11a_cross_system[_<tag>][_fast].csv
experiments/results/e11b_cross_system[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e11_scaling
    python -m experiments.e11_scaling --fast
    python -m experiments.e11_scaling --e11a
    python -m experiments.e11_scaling --e11b
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

from experiments import common
from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.models import registry
from src.simulation import simulator as simulator_lib

COLUMNS = ("scenario", "variant", "window", "mae", "rmse", "r2", "mape",
           "auroc", "n_samples")
# 4-class system-type codes matching the dataset encoder.
INFERENCE_CODES = {"A": 0.0, "B": 0.0, "C": 1.0, "D": 2.0, "E": 3.0}
# Published results the summary compares against: (name, scenario, R², AUROC).
BASELINES = (
    ("E8c physics_lstm", "D", 0.740, 0.789),
    ("E8c physics_lstm", "E", 0.575, 0.759),
    ("E10b Transformer w=30", "D", 0.734, 0.890),
    ("E10b Transformer w=30", "E", 0.567, 0.801),
)


def eval_per_scenario(
    predictor,
    simulator,
    groups,
    window: int,
    min_gap: int,
    scenarios: Sequence[str] = ("D", "E"),
    set_inference_codes: bool = False,
    eval_seed: int = 43,
) -> list[dict[str, object]]:
    """Scores a predictor on fresh trajectories of each scenario.

    Args:
        predictor: The trained predictor.
        simulator: The simulator.
        groups: Scenario label -> configurations.
        window: Observation window in time steps.
        min_gap: Minimum gap between window ends.
        scenarios: Scenario labels to score.
        set_inference_codes: Whether to set the predictor's inference
            interaction code per scenario (physics_lstm with context).
        eval_seed: Seed of the evaluation datasets.

    Returns:
        One row per scenario with the metrics.
    """
    generator = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store())
    rows = []
    for label in scenarios:
        if set_inference_codes and hasattr(
                predictor, "inference_interaction_code"):
            predictor.inference_interaction_code = INFERENCE_CODES.get(
                label, 0.0)
        dataset = generator.execute(generate_dataset.GenerateDatasetCommand(
            configs=groups[label],
            window_length=window,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=eval_seed,
            train_ratio=0.0, val_ratio=0.0,
            min_window_gap=min_gap,
        )).as_test_only()
        metrics = backtest.BacktestUseCase(predictor).execute(
            backtest.BacktestCommand(dataset=dataset, horizon=1.0))
        print(f"  [{label}] → {metrics.summary()}")
        rows.append(dict(
            scenario=label, mae=metrics.mae, rmse=metrics.rmse,
            r2=metrics.r2, mape=metrics.mape, auroc=metrics.risk_auroc,
            n_samples=metrics.n_samples,
        ))
    return rows


def _fit(build_predictor, groups, simulator, window: int, min_gap: int,
         n_epochs: int, seed: int, verbose: bool):
    """Builds a predictor and trains it on the D+E mixture.

    The predictor is built after the dataset, as in the published runs.
    """
    dataset = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store()
    ).execute(generate_dataset.GenerateDatasetCommand(
        configs=groups["D"] + groups["E"],
        window_length=window, horizon=1.0,
        samples_per_trajectory=5, seed=seed,
        train_ratio=0.9, val_ratio=0.1,
        min_window_gap=min_gap,
    ))
    print(f"  Train samples: {len(dataset.train_idx)}  "
          f"Val: {len(dataset.val_idx)}")
    predictor = build_predictor()
    train_model.TrainModelUseCase(predictor).execute(
        train_model.TrainModelCommand(
            dataset=dataset, n_epochs=n_epochs,
            batch_size=64, learning_rate=5e-4,
            regression_loss="huber", verbose=verbose,
            seed=seed,
        ))
    return predictor


def run_e11a(
    simulator,
    groups,
    n_epochs: int,
    n_per: int,
    verbose: bool = True,
    *,
    seed: int = 42,
    window: int = 20,
    eval_groups=None,
) -> list[dict[str, object]]:
    """E11a: Transformer on D+E with the E8c training settings.

    Defaults reproduce the published run; R3 and R4 vary the seed, the
    window and the evaluation configurations.

    Args:
        simulator: The simulator.
        groups: Scenario label -> training configurations.
        n_epochs: Training epochs.
        n_per: Trajectories per scenario (for the log line).
        verbose: Whether to print training progress.
        seed: Seed of the datasets and training; evaluation uses
            `seed + 1`.
        window: Observation window in time steps.
        eval_groups: Configurations to score on. The default is the
            training configurations, with fresh trajectories, as in the
            published run; R4 passes disjoint ones to measure what that
            is worth.

    Returns:
        One row per scenario, with the keys of `COLUMNS`.
    """
    min_gap = 10
    print(f"=== E11a: Transformer cross-system | window={window} | "
          f"n_per={n_per} ===")
    predictor = _fit(
        functools.partial(registry.build_model, "transformer_large"),
        groups, simulator, window, min_gap, n_epochs, seed, verbose)
    rows = eval_per_scenario(
        predictor, simulator, groups if eval_groups is None else eval_groups,
        window, min_gap, eval_seed=seed + 1)
    for row in rows:
        row["variant"] = f"transformer_w{window}"
        row["window"] = window
    return rows


def run_e11b(
    simulator,
    groups_1000,
    n_epochs: int,
    verbose: bool = True,
    *,
    seed: int = 42,
) -> list[dict[str, object]]:
    """E11b: physics + LSTM on D+E with window 20.

    Defaults reproduce the published run; the training-set size is chosen
    by the caller through `groups_1000`, which is what R3 varies.

    Args:
        simulator: The simulator.
        groups_1000: Scenario label -> configurations (1000 in the
            published run), used for training and scoring.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        seed: Seed of the datasets and training; evaluation uses
            `seed + 1`.

    Returns:
        One row per scenario, with the keys of `COLUMNS`.
    """
    window = 20
    min_gap = 15
    print(f"=== E11b: physics_lstm cross-system | 1000 trajs | "
          f"window={window} ===")
    build_predictor = functools.partial(
        registry.build_model,
        "physics_lstm",
        hidden_size=128, num_layers=2, dropout=0.3,
        adaptive_prior_r2_threshold=0.5,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=2.0,   # set per scenario when scoring
        inference_dissipator_code=0.0,
        early_stopping_patience=50,       # 3x dataset: more patience
    )
    predictor = _fit(
        build_predictor, groups_1000, simulator, window, min_gap, n_epochs,
        seed, verbose)
    rows = eval_per_scenario(
        predictor, simulator, groups_1000, window, min_gap,
        set_inference_codes=True, eval_seed=seed + 1)
    for row in rows:
        row["variant"] = "physics_lstm_1000"
        row["window"] = window
    return rows


def _add_flags(parser) -> None:
    parser.add_argument("--e11a", action="store_true", dest="only_a")
    parser.add_argument("--e11b", action="store_true", dest="only_b")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E11a and E11b and writes their CSVs."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, seeds=None, out=False)
    n_per_500, n_per_1000, epochs = (
        (50, 100, 10) if args.fast else (500, 1000, 150))
    config = {"n_per_500": n_per_500, "n_per_1000": n_per_1000,
              "n_epochs": epochs}
    simulator = simulator_lib.QuTipSimulator()
    groups_500 = scenario_lib.build_configs(n_per_scenario=n_per_500, seed=42)

    rows_a, rows_b = [], []
    if not args.only_b:
        print(f"\n{'=' * 70}\nE11a — Transformer window=20+E8c | "
              f"n_per={n_per_500} | epochs={epochs}\n{'=' * 70}")
        rows_a = run_e11a(simulator, groups_500, epochs, n_per_500)
        common.write_results(
            rows_a, common.results_path("e11a_cross_system", args),
            columns=COLUMNS, seeds=[42], config=config)

    if not args.only_a:
        print(f"\n{'=' * 70}\nE11b — physics_lstm 1000 trajs   | "
              f"n_per={n_per_1000} | epochs={epochs}\n{'=' * 70}")
        groups_1000 = scenario_lib.build_configs(
            n_per_scenario=n_per_1000, seed=42)
        rows_b = run_e11b(simulator, groups_1000, epochs)
        common.write_results(
            rows_b, common.results_path("e11b_cross_system", args),
            columns=COLUMNS, seeds=[42], config=config)

    print("\n\n=== SUMMARY vs baselines ===")
    print(f"\n{'Model':<28} {'Scen':>5} {'R²':>8} {'AUROC':>8}")
    print("-" * 55)
    for name, scenario, r2, auroc in BASELINES:
        print(f"  {name:<26} {scenario:>5} {r2:>8.4f} {auroc:>8.4f}")
    print()
    for row in rows_a + rows_b:
        print(f"  {row['variant']:<26} {row['scenario']:>5} "
              f"{row['r2']:>8.4f} {row['auroc']:>8.4f}")
    print("\nE11 completed. Results in experiments/results/e11*.csv")


if __name__ == "__main__":
    main()
