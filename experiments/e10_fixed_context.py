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
experiments/results/e10a_cross_system[_<tag>][_fast].csv
experiments/results/e10b_cross_system[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e10_fixed_context
    python -m experiments.e10_fixed_context --fast
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.models import registry
from src.simulation import simulator as simulator_lib

WINDOW_2Q = 30
MIN_GAP = 15
COLUMNS = (
    "name", "variant", "window_fraction", "noise_sigma", "mae", "rmse",
    "r2", "mape", "auroc", "n_samples")
# 4-class system-type codes (must match the dataset encoder).
SCENARIO_CODE = {"D": 2.0, "E": 3.0}
# Published baselines: scenario -> name -> (R², AUROC).
BASELINES = {
    "D": {"physics_lstm E8c": (0.740, 0.789),
          "Transformer E7a": (0.758, 0.863)},
    "E": {"physics_lstm E8c": (0.575, 0.759),
          "Transformer E7a": (0.599, 0.736)},
}


def _fit_on_mixture(
    generator: generate_dataset.GenerateDatasetUseCase,
    groups,
    predictor: ports.IPredictor,
    n_epochs: int,
    verbose: bool,
) -> None:
    """Trains the predictor on the D+E mixture with the E8c settings."""
    dataset = generator.execute(generate_dataset.GenerateDatasetCommand(
        configs=groups["D"] + groups["E"],
        window_length=WINDOW_2Q,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        train_ratio=0.9,
        val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))
    train_model.TrainModelUseCase(predictor).execute(
        train_model.TrainModelCommand(
            dataset=dataset,
            n_epochs=n_epochs,
            batch_size=64,
            learning_rate=5e-4,
            regression_loss="huber",
            verbose=verbose,
            seed=42,
        ))


def _score_per_scenario(
    generator: generate_dataset.GenerateDatasetUseCase,
    groups,
    predictor: ports.IPredictor,
    variant: str,
    set_codes: bool,
) -> list[dict[str, object]]:
    """Scores the predictor on fresh trajectories of D and of E."""
    rows = []
    for label in ("D", "E"):
        if set_codes:
            predictor.inference_interaction_code = SCENARIO_CODE[label]
        dataset = generator.execute(generate_dataset.GenerateDatasetCommand(
            configs=groups[label],
            window_length=WINDOW_2Q,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=43,
            train_ratio=0.0,
            val_ratio=0.0,
            min_window_gap=MIN_GAP,
        )).as_test_only()
        metrics = backtest.BacktestUseCase(predictor).execute(
            backtest.BacktestCommand(dataset=dataset, horizon=1.0))
        print(f"  [{label}] → {metrics.summary()}")
        rows.append({
            "name": f"cross_system_{label}",
            "variant": variant,
            "window_fraction": WINDOW_2Q,
            "noise_sigma": 0.0,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "r2": metrics.r2,
            "mape": metrics.mape,
            "auroc": metrics.risk_auroc,
            "n_samples": metrics.n_samples,
        })
    return rows


def run_e10a(simulator, groups, n_epochs: int,
             verbose: bool = True) -> list[dict[str, object]]:
    """E10a: physics + LSTM on D+E with the corrected system-type codes.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per scenario, with the keys of `COLUMNS`.
    """
    print("=== E10a: training physics_lstm on D+E mixture "
          "(fixed 4-class codes) ===")
    generator = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store())
    predictor = registry.build_model(
        "physics_lstm",
        hidden_size=128, num_layers=2, dropout=0.3,
        adaptive_prior_r2_threshold=0.5,
        risk_pos_weight=0.0, augment_sigma=0.02,
        # XXZ default; set per scenario when scoring.
        inference_interaction_code=2.0,
        inference_dissipator_code=0.0,
    )
    _fit_on_mixture(generator, groups, predictor, n_epochs, verbose)
    return _score_per_scenario(
        generator, groups, predictor, "physics_lstm", set_codes=True)


def run_e10b(simulator, groups, n_epochs: int,
             verbose: bool = True) -> list[dict[str, object]]:
    """E10b: Transformer on D+E with the E8c window and gap.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per scenario, with the keys of `COLUMNS`.
    """
    print("=== E10b: training Transformer on D+E mixture (E8c config) ===")
    generator = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store())
    predictor = registry.build_model("transformer_large")
    _fit_on_mixture(generator, groups, predictor, n_epochs, verbose)
    return _score_per_scenario(
        generator, groups, predictor, "transformer", set_codes=False)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E10a and E10b and writes their CSVs."""
    args = common.parse_args(__doc__, argv=argv, seeds=None, out=False)
    n_per, epochs = (50, 10) if args.fast else (500, 150)
    config = {"n_per": n_per, "n_epochs": epochs, "window": WINDOW_2Q,
              "min_gap": MIN_GAP}
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'=' * 70}\nE10a — Fixed context (physics_lstm) | "
          f"n_per={n_per} | epochs={epochs}\n{'=' * 70}")
    rows_a = run_e10a(simulator, groups, epochs)
    common.write_results(
        rows_a, common.results_path("e10a_cross_system", args),
        columns=COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE10b — Transformer cross-system | "
          f"n_per={n_per} | epochs={epochs}\n{'=' * 70}")
    rows_b = run_e10b(simulator, groups, epochs)
    common.write_results(
        rows_b, common.results_path("e10b_cross_system", args),
        columns=COLUMNS, seeds=[42], config=config)

    print("\n\n=== SUMMARY ===")
    print(f"\n{'Model':<20} {'Scenario':<10} {'R²':>8} {'AUROC':>8}")
    print("-" * 50)
    for row in rows_a + rows_b:
        scenario = row["name"].split("_")[-1]
        print(f"{row['variant']:<20} {scenario:<10} {row['r2']:>8.4f} "
              f"{row['auroc']:>8.4f}")
    print("\nBaselines:")
    for scenario, models in BASELINES.items():
        for name, (r2, auroc) in models.items():
            print(f"  {name:<20} {scenario:<10} {r2:>8.4f} {auroc:>8.4f}")
    print("\nE10 completed. Results in experiments/results/e10*.csv")


if __name__ == "__main__":
    main()
