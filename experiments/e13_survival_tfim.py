"""Experiment E13 — Survival loss vs dropping censored TFIM trajectories.

Goal
----
Compare keep+SurvHuber against the drop-censored pipeline from task 06
on TFIM. SurvHuber should not hurt uncensored-test R² and should use
more windows when T₂ > t_max is common.

Claim
-----
C2: TFIM needs correct labels, not only ML. Right-censored T₂ must not
    be treated as t_max and should not be dropped without a survival loss.

Scenarios
---------
E (2q TFIM). window=30, hidden=128, τ=0, E8 regularisation.
drop: include_censored=False, Huber. surv: include_censored=True, SurvHuber.

Metrics
-------
R² on uncensored test windows only. At t_max=20 censoring is rare
(~5 windows / 200 trajs); no gain until γ/t_max produce more T₂ > t_max.

Output
------
experiments/results/e13_survival_tfim[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e13_survival_tfim
    python -m experiments.e13_survival_tfim --fast
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.domain import value_objects
from src.models import registry
from src.simulation import simulator as simulator_lib

COLUMNS = (
    "name", "variant", "scenario", "include_censored", "loss", "mae",
    "rmse", "r2", "mape", "auroc", "n_samples", "n_censored_traj",
    "n_censored_windows", "n_train")


def _lstm() -> ports.IPredictor:
    return registry.build_model(
        "physics_lstm", hidden_size=128, num_layers=2, dropout=0.3,
        adaptive_prior_r2_threshold=0.0,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=3.0,
        early_stopping_patience=30,
    )


def run_arm(simulator, configs, *, include_censored: bool, loss: str,
            n_epochs: int, window: int,
            verbose: bool) -> dict[str, object]:
    """Trains and scores one arm of the comparison.

    Args:
        simulator: The simulator.
        configs: TFIM configurations.
        include_censored: Whether right-censored trajectories are kept.
        loss: Regression loss alias.
        n_epochs: Training epochs.
        window: Observation window in time steps.
        verbose: Whether to print training progress.

    Returns:
        One row with the keys of `COLUMNS`.
    """
    dataset = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store()
    ).execute(generate_dataset.GenerateDatasetCommand(
        configs=configs,
        window_length=window,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        min_window_gap=max(window // 2, 1),
        include_censored=include_censored,
    ))
    predictor = _lstm()
    train_model.TrainModelUseCase(predictor).execute(
        train_model.TrainModelCommand(
            dataset=dataset,
            n_epochs=n_epochs,
            batch_size=64,
            learning_rate=5e-4,
            regression_loss=loss,
            verbose=verbose,
            seed=42,
        ))
    metrics = backtest.BacktestUseCase(predictor).execute(
        backtest.BacktestCommand(dataset=dataset, horizon=1.0))
    n_censored_windows = int(dataset.metadata.get("n_censored_windows", 0))
    print(f"  → {metrics.summary()}  censored_windows={n_censored_windows}")
    return {
        "name": "surv" if include_censored else "drop",
        "variant": value_objects.PredictorVariant.PHYSICS_LSTM.value,
        "scenario": "E",
        "include_censored": include_censored,
        "loss": loss,
        "mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2,
        "mape": metrics.mape, "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
        "n_censored_traj": dataset.metadata.get("n_censored", 0),
        "n_censored_windows": n_censored_windows,
        "n_train": dataset.metadata.get("n_train", 0),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Runs both arms and writes the CSV."""
    args = common.parse_args(__doc__, argv=argv, seeds=None)
    n_per, epochs, window = (40, 15, 20) if args.fast else (200, 80, 30)
    configs = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)["E"]
    simulator = simulator_lib.QuTipSimulator()

    print(f"E13 survival TFIM | n_per={n_per} | epochs={epochs} | "
          f"window={window}")
    rows = []
    print("\n=== drop censored + Huber ===")
    rows.append(run_arm(
        simulator, configs, include_censored=False, loss="huber",
        n_epochs=epochs, window=window, verbose=True))
    print("\n=== keep censored + SurvHuber ===")
    rows.append(run_arm(
        simulator, configs, include_censored=True, loss="surv_huber",
        n_epochs=epochs, window=window, verbose=True))

    path = common.results_path("e13_survival_tfim", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"n_per": n_per, "n_epochs": epochs, "window": window})
    print(f"Saved → {path}")
    print(f"\n{'arm':<8} {'R²':>8} {'n_test':>8} {'n_train':>8} "
          f"{'cens_win':>8}")
    for row in rows:
        print(f"{row['name']:<8} {row['r2']:>8.4f} {row['n_samples']:>8} "
              f"{row['n_train']:>8} {row['n_censored_windows']:>8}")


if __name__ == "__main__":
    main()
