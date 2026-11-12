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
experiments/results/e14_inject_J[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e14_inject_J
    python -m experiments.e14_inject_J --fast
    python -m experiments.e14_inject_J --e14a
    python -m experiments.e14_inject_J --e14b
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments import common
from src.application import backtest as backtest_use_case
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.evaluation import backtest
from src.evaluation import metrics
from src.models import registry
from src.simulation import simulator as simulator_lib

WINDOW = 30
MIN_GAP = 15
COLUMNS = ("arm", "scenario", "use_J", "j_bin", "r2", "mae", "n_samples",
           "auroc")
J_BINS = (
    ("J_lt_0.5", 0.0, 0.5),
    ("J_0.5_0.8", 0.5, 0.8),
    ("J_ge_0.8", 0.8, 1.01),
)


def _lstm(
    *, use_j: bool, tau: float, interaction_code: float
) -> ports.IPredictor:
    return registry.build_model(
        "physics_lstm", hidden_size=128, num_layers=2, dropout=0.3,
        adaptive_prior_r2_threshold=tau,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=interaction_code,
        use_J_scalar=use_j,
        early_stopping_patience=30,
    )


def _predict_all(predictor, dataset, horizon: float):
    """Predicts every test window, feeding J when the dataset has it."""
    has_j = len(dataset.J_values) == len(dataset.sequences)
    predictions, actuals, couplings = [], [], []
    for i in backtest.test_indices(dataset):
        if has_j:
            predictor.inference_J = float(dataset.J_values[i])
        out = predictor.predict(
            dataset.sequences[i], float(dataset.t_obs[i]), horizon)
        predictions.append(out.t_decoh_predicted)
        actuals.append(float(dataset.t_decoh_abs[i]))
        couplings.append(float(dataset.J_values[i]) if has_j else 0.0)
    return (np.asarray(predictions), np.asarray(actuals),
            np.asarray(couplings))


def _bin_rows(arm: str, scenario: str, use_j: bool, predictions, actuals,
              couplings) -> list[dict[str, object]]:
    rows = []
    for name, low, high in J_BINS:
        mask = (couplings >= low) & (couplings < high)
        n = int(mask.sum())
        rows.append({
            "arm": arm,
            "scenario": scenario,
            "use_J": use_j,
            "j_bin": name,
            "r2": (metrics.r2(actuals[mask], predictions[mask]) if n >= 3
                   else float("nan")),
            "mae": (metrics.mae(actuals[mask], predictions[mask]) if n
                    else float("nan")),
            "n_samples": n,
            "auroc": float("nan"),
        })
    return rows


def _train(simulator, configs, *, use_j: bool, tau: float, n_epochs: int,
           verbose: bool, interaction_code: float) -> ports.IPredictor:
    dataset = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store()
    ).execute(generate_dataset.GenerateDatasetCommand(
        configs=configs,
        window_length=WINDOW,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=42,
        train_ratio=0.9,
        val_ratio=0.1,
        min_window_gap=MIN_GAP,
    ))
    predictor = _lstm(
        use_j=use_j, tau=tau, interaction_code=interaction_code)
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
    return predictor


def _eval_group(simulator, predictor, configs, *, arm: str, scenario: str,
                use_j: bool, seed: int) -> list[dict[str, object]]:
    """Scores overall and per J bin on fresh trajectories."""
    dataset = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store()
    ).execute(generate_dataset.GenerateDatasetCommand(
        configs=configs,
        window_length=WINDOW,
        horizon=1.0,
        samples_per_trajectory=5,
        seed=seed,
        train_ratio=0.0,
        val_ratio=0.0,
        min_window_gap=MIN_GAP,
    )).as_test_only()
    scored = backtest_use_case.BacktestUseCase(predictor).execute(
        backtest_use_case.BacktestCommand(dataset=dataset, horizon=1.0))
    predictions, actuals, couplings = _predict_all(predictor, dataset, 1.0)
    print(f"  {arm} {scenario} use_J={use_j} → {scored.summary()}")
    overall = {
        "arm": arm,
        "scenario": scenario,
        "use_J": use_j,
        "j_bin": "all",
        "r2": scored.r2,
        "mae": scored.mae,
        "n_samples": scored.n_samples,
        "auroc": scored.risk_auroc,
    }
    return [overall] + _bin_rows(
        arm, scenario, use_j, predictions, actuals, couplings)


def run_e14a(simulator, groups, n_epochs: int,
             verbose: bool) -> list[dict[str, object]]:
    """E14a: TFIM-only model with and without J.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        Overall and per-J-bin rows of both arms, with the keys of
        `COLUMNS`.
    """
    rows = []
    for use_j in (False, True):
        label = "e14a_J" if use_j else "e14a_noJ"
        print(f"\n=== {label} — TFIM-only ===")
        predictor = _train(
            simulator, groups["E"], use_j=use_j, tau=0.0,
            n_epochs=n_epochs, verbose=verbose, interaction_code=3.0)
        rows.extend(_eval_group(
            simulator, predictor, groups["E"],
            arm=label, scenario="E", use_j=use_j, seed=43))
    return rows


def run_e14b(simulator, groups, n_epochs: int,
             verbose: bool) -> list[dict[str, object]]:
    """E14b: one model with J trained on D+E, scored per scenario.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        Overall and per-J-bin rows of both scenarios, with the keys of
        `COLUMNS`.
    """
    rows = []
    print("\n=== e14b_J — cross-system D+E ===")
    predictor = _train(
        simulator, groups["D"] + groups["E"], use_j=True, tau=0.5,
        n_epochs=n_epochs, verbose=verbose, interaction_code=2.5)
    for scenario, code, configs in (
            ("D", 2.0, groups["D"]), ("E", 3.0, groups["E"])):
        predictor.inference_interaction_code = code
        rows.extend(_eval_group(
            simulator, predictor, configs,
            arm="e14b_J", scenario=scenario, use_j=True, seed=43))
    return rows


def _add_flags(parser) -> None:
    parser.add_argument("--e14a", action="store_true")
    parser.add_argument("--e14b", action="store_true")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E14a and/or E14b and writes the CSV."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, seeds=None)
    n_per, epochs = (40, 12) if args.fast else (500, 150)
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)
    simulator = simulator_lib.QuTipSimulator()

    print(f"E14 inject J | n_per={n_per} | epochs={epochs} | "
          f"window={WINDOW}")
    rows = []
    if args.e14a or not args.e14b:
        rows.extend(run_e14a(simulator, groups, epochs, verbose=True))
    if args.e14b or not args.e14a:
        rows.extend(run_e14b(simulator, groups, epochs, verbose=True))

    path = common.results_path("e14_inject_J", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"n_per": n_per, "n_epochs": epochs, "window": WINDOW})
    print(f"Saved → {path}")

    print(f"\n{'arm':<12} {'sc':<3} {'bin':<12} {'R²':>8} {'N':>6}")
    print("-" * 46)
    for row in rows:
        r2 = row["r2"]
        r2_text = f"{r2:8.4f}" if r2 == r2 else f"{'nan':>8}"
        print(f"{row['arm']:<12} {row['scenario']:<3} {row['j_bin']:<12} "
              f"{r2_text} {row['n_samples']:>6}")


if __name__ == "__main__":
    main()
