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
experiments/results/e6_ablation_improved[_<tag>][_fast].csv
experiments/results/e6_noise_sweep_improved[_<tag>][_fast].csv
experiments/results/e6_cross_system_improved[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e6_2qubit_improved
    python -m experiments.e6_2qubit_improved --fast
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

ABLATION_COLUMNS = (
    "scenario", "name", "variant", "window_fraction", "noise_sigma",
    "mae", "rmse", "r2", "mape", "auroc", "n_samples")
RESULT_COLUMNS = (
    "name", "variant", "window_fraction", "noise_sigma", "mae", "rmse",
    "r2", "mape", "auroc", "n_samples")
ABLATION_VARIANTS = (
    value_objects.PredictorVariant.PHYSICS_ONLY,
    value_objects.PredictorVariant.LSTM_ONLY,
    value_objects.PredictorVariant.PHYSICS_LSTM,
    value_objects.PredictorVariant.TRANSFORMER,
)
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)

# The improved architecture: larger hidden size, two layers, an adaptive
# prior that is gated off when the OLS fit of log C is poor.
_LSTM_SIZE = {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}
improved_lstm = functools.partial(
    registry.build_model, "physics_lstm", **_LSTM_SIZE,
    adaptive_prior_r2_threshold=0.7)


def improved_ablation_predictor(
    variant: value_objects.PredictorVariant,
    command: run_ablation.AblationCommand,
) -> ports.IPredictor:
    """Builds the predictor of one ablation variant of E6a.

    The base use case builds weaker defaults (hidden 64, one layer, tau
    0), which is not the model the paper reports; R1 uses this factory so
    that it replicates E6a's architecture.

    Args:
        variant: The ablation variant.
        command: The ablation command.

    Returns:
        The untrained predictor.
    """
    if variant == value_objects.PredictorVariant.LSTM_ONLY:
        return registry.build_model(
            "lstm_only", **_LSTM_SIZE, adaptive_prior_r2_threshold=0.0)
    if variant == value_objects.PredictorVariant.PHYSICS_LSTM:
        return improved_lstm()
    return run_ablation.default_predictor(variant, command)


def run_e6a(
    simulator,
    groups,
    scenarios: Sequence[str],
    n_epochs: int,
    *,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """E6a: ablation of four variants on each two-qubit scenario.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        scenarios: Scenario labels to run.
        n_epochs: Training epochs.
        seed: Seed of the dataset and training.
        verbose: Whether to print training progress.

    Returns:
        One row per scenario and variant, with the keys of
        `ABLATION_COLUMNS`.
    """
    use_case = run_ablation.AblationStudyUseCase(
        simulator, store=scenario_lib.trajectory_store(),
        predictor_factory=improved_ablation_predictor)
    rows = []
    for label in scenarios:
        print(f"\n{'=' * 60}\nScenario {label}\n{'=' * 60}")
        results = use_case.execute(run_ablation.AblationCommand(
            configs=groups[label],
            variants=list(ABLATION_VARIANTS),
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=n_epochs,
            batch_size=64,
            lr=5e-4,          # lower learning rate for the larger model
            regression_loss="huber",
            seed=seed,
            verbose=verbose,
        ))
        for result in results:
            rows.append({**result.as_dict(), "scenario": label})
    return rows


def run_e6b(
    simulator,
    groups,
    scenarios: Sequence[str],
    n_epochs: int,
    *,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """E6b: noise robustness of the improved model on pooled scenarios.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        scenarios: Scenario labels pooled into one dataset.
        n_epochs: Training epochs.
        seed: Seed of the dataset, training and noise.
        verbose: Whether to print training progress.

    Returns:
        One row per sigma, with the keys of `RESULT_COLUMNS`.
    """
    configs = [cfg for label in scenarios for cfg in groups[label]]
    # No trajectory cache here: the published run did not use one.
    results = run_noise_sweep.NoiseSweepUseCase(
        simulator, predictor_factory=improved_lstm
    ).execute(run_noise_sweep.NoiseSweepCommand(
        configs=configs,
        sigmas=list(SIGMAS),
        window_length=50,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=seed,
        verbose=verbose,
    ))
    return [result.as_dict() for result in results]


def run_e6c(
    simulator,
    groups,
    train_scenarios: Sequence[str],
    eval_scenarios: Sequence[str],
    n_epochs: int,
    *,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """E6c: cross-system generalisation of the improved model.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        train_scenarios: Scenario labels of the training mixture.
        eval_scenarios: Scenario labels scored separately.
        n_epochs: Training epochs.
        seed: Seed of the datasets and training.
        verbose: Whether to print training progress.

    Returns:
        One row per evaluation scenario, with the keys of
        `RESULT_COLUMNS`.
    """
    results = run_cross_system.CrossSystemUseCase(
        simulator, store=scenario_lib.trajectory_store(),
        predictor_factory=improved_lstm
    ).execute(run_cross_system.CrossSystemCommand(
        train_configs=[cfg for label in train_scenarios
                       for cfg in groups[label]],
        eval_groups={label: groups[label] for label in eval_scenarios},
        window_length=50,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=seed,
        verbose=verbose,
    ))
    return [result.as_dict() for result in results]


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E6a, E6b and E6c and writes their CSVs."""
    args = common.parse_args(__doc__, argv=argv, seeds=None, out=False)
    n_per, epochs = (50, 10) if args.fast else (500, 150)
    config = {"n_per": n_per, "n_epochs": epochs}
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)
    scenarios = ["D", "E"]

    print(f"\n{'=' * 70}\nE6a — Ablation (D, E) | n_per={n_per} | "
          f"epochs={epochs}\n{'=' * 70}")
    rows = run_e6a(simulator, groups, scenarios, epochs)
    common.write_results(
        rows, common.results_path("e6_ablation_improved", args),
        columns=ABLATION_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE6b — Noise sweep (D+E) | n_per={n_per} | "
          f"epochs={epochs}\n{'=' * 70}")
    rows = run_e6b(simulator, groups, scenarios, epochs)
    common.write_results(
        rows, common.results_path("e6_noise_sweep_improved", args),
        columns=RESULT_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE6c — Cross-system (train D+E, eval D/E) | "
          f"n_per={n_per} | epochs={epochs}\n{'=' * 70}")
    rows = run_e6c(simulator, groups, scenarios, scenarios, epochs)
    common.write_results(
        rows, common.results_path("e6_cross_system_improved", args),
        columns=RESULT_COLUMNS, seeds=[42], config=config)
    print("\nE6 completed. Results in experiments/results/e6_*_improved*.csv")


if __name__ == "__main__":
    main()
