"""Experiment E8 — Improved two-qubit training with E7 findings applied.

Goal
----
Apply E7b τ and four training fixes to physics_lstm; keep the best
valid TFIM cross-system number (E8c R²=0.575).

Claim
-----
C2/C5: per-scenario τ (D=0.5, E=0.0) + weighted BCE + noise aug
+ window=30 + min_gap=15 beat E6c on D and E.

Scenarios
---------
D (2q XXZ), E (2q TFIM).
E8a ablation, E8b noise, E8c cross-system (train D+E).

Metrics
-------
R², MAE, AUROC. Paper 2 baseline: E8c D=0.740, E=0.575.

Output
------
experiments/results/e8a_ablation[_<tag>][_fast].csv
experiments/results/e8b_noise_sweep[_<tag>][_fast].csv
experiments/results/e8c_cross_system[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e8_improved_2qubit
    python -m experiments.e8_improved_2qubit --fast
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

from experiments import common
from src.application import run_ablation
from src.application import run_cross_system
from src.application import run_noise_sweep
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.domain import value_objects
from src.models import registry
from src.simulation import simulator as simulator_lib

WINDOW_2Q = 30       # wider window for 2-qubit physics slope stability
MIN_GAP = 15         # min window gap (half of WINDOW_2Q)
# Per-scenario τ (from the E7b results).
TAU_PER_SCENARIO = {"D": 0.5, "E": 0.0}
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)

ABLATION_COLUMNS = (
    "scenario", "tau", "name", "variant", "window_fraction", "noise_sigma",
    "mae", "rmse", "r2", "mape", "auroc", "n_samples")
RESULT_COLUMNS = (
    "name", "variant", "window_fraction", "noise_sigma", "mae", "rmse",
    "r2", "mape", "auroc", "n_samples")

# Training choices of E8: no weighted BCE, small input noise augmentation.
_LSTM_SIZE = {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}
_TRAINING = {"risk_pos_weight": 0.0, "augment_sigma": 0.02}


def _improved_lstm(tau: float) -> ports.IPredictor:
    return registry.build_model(
        "physics_lstm", **_LSTM_SIZE, adaptive_prior_r2_threshold=tau,
        **_TRAINING)


def _ablation_predictor(
    variant: value_objects.PredictorVariant,
    command: run_ablation.AblationCommand,
    *,
    tau: float,
) -> ports.IPredictor:
    """Builds the predictor of one E8a variant for a scenario's τ."""
    variants = value_objects.PredictorVariant
    if variant == variants.PHYSICS_ONLY:
        return run_ablation.default_predictor(variant, command)
    if variant == variants.LSTM_ONLY:
        return registry.build_model("lstm_only", **_LSTM_SIZE, **_TRAINING)
    if variant == variants.PHYSICS_LSTM:
        return _improved_lstm(tau)
    if variant == variants.TRANSFORMER:
        return registry.build_model("transformer_large")
    raise ValueError(variant)


def run_e8a(simulator, groups, n_epochs: int,
            verbose: bool = True) -> list[dict[str, object]]:
    """E8a: ablation of every variant on D and on E.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per scenario and variant, with the keys of
        `ABLATION_COLUMNS`.
    """
    rows = []
    for label in ("D", "E"):
        tau = TAU_PER_SCENARIO[label]
        print(f"\n{'=' * 60}\nE8a — Scenario {label}  (τ={tau})\n"
              f"{'=' * 60}")
        results = run_ablation.AblationStudyUseCase(
            simulator, store=scenario_lib.trajectory_store(),
            predictor_factory=functools.partial(
                _ablation_predictor, tau=tau),
        ).execute(run_ablation.AblationCommand(
            configs=groups[label],
            variants=list(value_objects.PredictorVariant),
            window_length=WINDOW_2Q,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=n_epochs,
            batch_size=64,
            lr=5e-4,
            regression_loss="huber",
            seed=42,
            verbose=verbose,
            min_window_gap=MIN_GAP,
        ))
        for result in results:
            rows.append({**result.as_dict(), "scenario": label, "tau": tau})

    print(f"\n{'Scenario':<8} {'τ':>5} {'Variant':<16} {'R²':>8} {'MAE':>8} "
          f"{'AUROC':>8}")
    print("-" * 58)
    for row in rows:
        print(f"{row['scenario']:<8} {row['tau']:>5.1f} "
              f"{row['variant']:<16} {row['r2']:>8.4f} {row['mae']:>8.4f} "
              f"{row['auroc']:>8.4f}")
    return rows


def run_e8b(simulator, groups, n_epochs: int, verbose: bool = True,
            *, seed: int = 42) -> list[dict[str, object]]:
    """E8b: noise robustness of the improved model on the D+E mixture.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        seed: Seed of the dataset, training and noise.

    Returns:
        One row per sigma, with the keys of `RESULT_COLUMNS`.
    """
    results = run_noise_sweep.NoiseSweepUseCase(
        simulator,
        # A compromise between τ of D (0.5) and of E (0.0).
        predictor_factory=functools.partial(_improved_lstm, 0.5),
    ).execute(run_noise_sweep.NoiseSweepCommand(
        configs=groups["D"] + groups["E"],
        sigmas=list(SIGMAS),
        window_length=WINDOW_2Q,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=seed,
        verbose=verbose,
        min_window_gap=MIN_GAP,
    ))
    rows = [result.as_dict() for result in results]

    print(f"\n{'σ_noise':>9} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 38)
    for row in rows:
        flag = " ✓" if row["auroc"] >= 0.90 else " ✗"
        print(f"{row['noise_sigma']:>9.3f} {row['r2']:>8.4f} "
              f"{row['mae']:>8.4f} {row['auroc']:>8.4f}{flag}")
    return rows


def run_e8c(simulator, groups, n_epochs: int, verbose: bool = True,
            *, seed: int = 42) -> list[dict[str, object]]:
    """E8c: improved physics + LSTM trained on D+E, scored per scenario.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.
        seed: Seed of the configurations' datasets and of training.

    Returns:
        One row per evaluation scenario, with the keys of
        `RESULT_COLUMNS`.
    """
    results = run_cross_system.CrossSystemUseCase(
        simulator, store=scenario_lib.trajectory_store(),
        predictor_factory=functools.partial(_improved_lstm, 0.5),
    ).execute(run_cross_system.CrossSystemCommand(
        train_configs=groups["D"] + groups["E"],
        eval_groups={"D": groups["D"], "E": groups["E"]},
        window_length=WINDOW_2Q,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=42,
        verbose=verbose,
        min_window_gap=MIN_GAP,
    ))
    rows = [result.as_dict() for result in results]

    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for row in rows:
        print(f"{row['name']:>20} {row['r2']:>8.4f} {row['mae']:>8.4f} "
              f"{row['auroc']:>8.4f} {row['n_samples']:>6}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E8a, E8b and E8c and writes their CSVs."""
    args = common.parse_args(__doc__, argv=argv, seeds=None, out=False)
    n_per, epochs = (50, 10) if args.fast else (500, 150)
    config = {"n_per": n_per, "n_epochs": epochs, "window": WINDOW_2Q,
              "min_gap": MIN_GAP}
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'=' * 70}\nE8a — Ablation (D+E) | n_per={n_per} | "
          f"epochs={epochs} | window={WINDOW_2Q}\n{'=' * 70}")
    rows = run_e8a(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e8a_ablation", args),
        columns=ABLATION_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE8b — Noise sweep (D+E) | n_per={n_per} | "
          f"epochs={epochs}\n{'=' * 70}")
    rows = run_e8b(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e8b_noise_sweep", args),
        columns=RESULT_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE8c — Cross-system (D+E) | n_per={n_per} | "
          f"epochs={epochs}\n{'=' * 70}")
    rows = run_e8c(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e8c_cross_system", args),
        columns=RESULT_COLUMNS, seeds=[42], config=config)
    print("\nE8 completed. Results in experiments/results/e8*.csv")


if __name__ == "__main__":
    main()
