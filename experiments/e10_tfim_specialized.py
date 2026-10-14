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
experiments/results/e10_tfim_a|b|c[_<tag>][_fast].csv
(a and b were R²<0 and are not kept; the catalog cites e10_tfim_c.csv)

Usage
-----
    python -m experiments.e10_tfim_specialized
    python -m experiments.e10_tfim_specialized --fast

E10a updates the best TFIM checkpoint in checkpoints/ when it beats the
stored one.
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

from experiments import common
from src.application import backtest
from src.application import generate_dataset
from src.application import run_cross_system
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.domain import value_objects
from src.models import registry
from src.persistence import stores
from src.simulation import simulator as simulator_lib

SINGLE_COLUMNS = (
    "name", "variant", "scenario", "mae", "rmse", "r2", "mape", "auroc",
    "n_samples")
CROSS_COLUMNS = (
    "name", "variant", "scenario", "window_fraction", "noise_sigma", "mae",
    "rmse", "r2", "mape", "auroc", "n_samples")


def _make_tfim_lstm(
    hidden: int, tau: float, int_code: float
) -> ports.IPredictor:
    return registry.build_model(
        "physics_lstm",
        hidden_size=hidden,
        num_layers=2,
        dropout=0.3,
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
    """Trains a TFIM-only model and backtests it on its own dataset."""
    dataset = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store()
    ).execute(generate_dataset.GenerateDatasetCommand(
        configs=configs,
        window_length=window,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        min_window_gap=min_gap,
    ))
    predictor = _make_tfim_lstm(hidden, tau, int_code)
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
    metrics = backtest.BacktestUseCase(predictor).execute(
        backtest.BacktestCommand(dataset=dataset, horizon=1.0))
    return metrics, predictor


def _single_row(name: str, metrics) -> dict[str, object]:
    return {
        "name": name,
        "variant": value_objects.PredictorVariant.PHYSICS_LSTM.value,
        "scenario": "E",
        "mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2,
        "mape": metrics.mape, "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
    }


def _update_registry(predictor, metrics) -> None:
    """Keeps the predictor as the best TFIM model if it beats the stored."""
    updated = stores.HamiltonianModelRegistry().maybe_update(
        predictor, metrics, 2, value_objects.InteractionType.TFIM,
        value_objects.DissipatorType.SIGMA_MINUS)
    print(f"  registry best_TFIM.pt updated={updated}")


def run_e10a(simulator, groups, n_epochs: int, verbose: bool = True,
             *, fast: bool = False) -> list[dict[str, object]]:
    """E10a: TFIM-only model, window 40.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        fast: Whether to use the smoke-run window and hidden size.

    Returns:
        One row, with the keys of `SINGLE_COLUMNS`.
    """
    window = 20 if fast else 40
    hidden = 64 if fast else 256
    print(f"\n{'=' * 60}\nE10a — TFIM-only  window={window}  "
          f"hidden={hidden}  τ=0\n{'=' * 60}")
    metrics, predictor = _train_single(
        simulator, groups["E"], window=window, hidden=hidden, tau=0.0,
        int_code=3.0, n_epochs=n_epochs, min_gap=max(window // 2, 1),
        verbose=verbose)
    print(f"  → {metrics.summary()}")
    _update_registry(predictor, metrics)
    return [_single_row("e10a_tfim_only", metrics)]


def run_e10b(simulator, groups, n_epochs: int, verbose: bool = True,
             *, fast: bool = False) -> list[dict[str, object]]:
    """E10b: TFIM-only model, window 50.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        fast: Whether to use the smoke-run window and hidden size.

    Returns:
        One row, with the keys of `SINGLE_COLUMNS`.
    """
    window = 25 if fast else 50
    hidden = 64 if fast else 256
    print(f"\n{'=' * 60}\nE10b — TFIM-only  window={window}  "
          f"hidden={hidden}  τ=0  code=3\n{'=' * 60}")
    metrics, _ = _train_single(
        simulator, groups["E"], window=window, hidden=hidden, tau=0.0,
        int_code=3.0, n_epochs=n_epochs, min_gap=max(window // 2, 1),
        verbose=verbose)
    print(f"  → {metrics.summary()}")
    return [_single_row("e10b_tfim_window50", metrics)]


def run_e10c(simulator, groups, n_epochs: int, verbose: bool = True,
             *, fast: bool = False) -> list[dict[str, object]]:
    """E10c: one mixed model trained on D+E, scored per scenario.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        fast: Whether to use the smoke-run window and hidden size.

    Returns:
        One row per scenario, with the keys of `CROSS_COLUMNS`.
    """
    window = 20 if fast else 40
    hidden = 64 if fast else 256
    print(f"\n{'=' * 60}\nE10c — cross-system D+E  window={window}  "
          f"hidden={hidden}\n{'=' * 60}")
    # No trajectory cache here: the published run did not use one.
    results = run_cross_system.CrossSystemUseCase(
        simulator,
        predictor_factory=functools.partial(
            _make_tfim_lstm, hidden, 0.0, 2.5),
    ).execute(run_cross_system.CrossSystemCommand(
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
    ))
    rows = [{**r.as_dict(),
             "scenario": r.spec.name.replace("cross_system_", "")}
            for r in results]
    print(f"\n{'Scenario':>8} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    for result in results:
        print(f"{result.spec.name:>8} {result.r2:>8.4f} "
              f"{result.mae:>8.4f} {result.risk_auroc:>8.4f}")
    return rows


def run_e10d(simulator, groups, n_epochs: int, verbose: bool = True,
             *, fast: bool = False) -> list[dict[str, object]]:
    """E10d: TFIM-only with the E8 window (30), keeping the test split.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        fast: Whether to use the smoke-run window and hidden size.

    Returns:
        One row, with the keys of `SINGLE_COLUMNS`.
    """
    window = 20 if fast else 30
    hidden = 64 if fast else 256
    print(f"\n{'=' * 60}\nE10d — TFIM-only  window={window}  "
          f"hidden={hidden}  τ=0\n{'=' * 60}")
    metrics, predictor = _train_single(
        simulator, groups["E"], window=window, hidden=hidden, tau=0.0,
        int_code=3.0, n_epochs=n_epochs, min_gap=max(window // 2, 1),
        verbose=verbose)
    print(f"  → {metrics.summary()}")
    _update_registry(predictor, metrics)
    return [_single_row("e10d_tfim_window30", metrics)]


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E10a, E10b and E10c and writes their CSVs."""
    args = common.parse_args(__doc__, argv=argv, seeds=None, out=False)
    n_per, epochs = (40, 15) if args.fast else (300, 120)
    config = {"n_per": n_per, "n_epochs": epochs}
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)

    print(f"E10 TFIM specialized | n_per={n_per} | epochs={epochs}")
    for stem, run, columns in (
        ("e10_tfim_a", run_e10a, SINGLE_COLUMNS),
        ("e10_tfim_b", run_e10b, SINGLE_COLUMNS),
        ("e10_tfim_c", run_e10c, CROSS_COLUMNS),
    ):
        rows = run(simulator, groups, epochs, fast=args.fast)
        common.write_results(
            rows, common.results_path(stem, args), columns=columns,
            seeds=[42], config=config)
    print("\nE10 completed. Results in experiments/results/e10_tfim_*.csv")


if __name__ == "__main__":
    main()
