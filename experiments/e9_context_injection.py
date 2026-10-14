"""Experiment E9 — Context-aware 2-qubit prediction.

Goal
----
Test whether injecting system identity (4-class code + jump-op code + OLS R²)
raises cross-system R² vs the E8c blind model.

Claim
-----
C5: a model that sees which Hamiltonian/dissipator it is in generalises
better than a context-blind residual LSTM.

Scenarios
---------
D (code 2), E (code 3). Jump-op: σ₋=0 / σz=1.
E9a: train D+E, eval D/E. E9b: context vs no-context per scenario.

Metrics
-------
R², MAE, AUROC vs E8c (D=0.740, E=0.575). After code fix: D=0.537, E=0.452.

Output
------
experiments/results/e9a_cross_system[_<tag>][_fast].csv
experiments/results/e9b_ablation[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e9_context_injection
    python -m experiments.e9_context_injection --fast
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

from experiments import common
from src.application import run_ablation
from src.application import run_cross_system
from src.data import scenarios as scenario_lib
from src.domain import ports
from src.domain import value_objects
from src.models import registry
from src.simulation import simulator as simulator_lib

WINDOW_2Q = 30
MIN_GAP = 15
E9A_COLUMNS = (
    "name", "variant", "window_fraction", "noise_sigma", "mae", "rmse",
    "r2", "mape", "auroc", "n_samples")
E9B_COLUMNS = (
    "scenario", "context_code", "name", "variant", "mae", "rmse", "r2",
    "mape", "auroc", "n_samples")


def make_context_predictor(
    interaction_code: float = 0.5,
    dissipator_code: float = 0.0,
    tau: float = 0.5,
) -> ports.IPredictor:
    """Builds the context-aware physics + LSTM predictor.

    Args:
        interaction_code: Inference-time system-type code.
        dissipator_code: Inference-time jump-operator code.
        tau: Adaptive-prior threshold on the OLS R² of log C.

    Returns:
        The untrained predictor.
    """
    return registry.build_model(
        "physics_lstm",
        hidden_size=128, num_layers=2, dropout=0.3,
        adaptive_prior_r2_threshold=tau,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=interaction_code,
        inference_dissipator_code=dissipator_code,
    )


def _ablation_predictor(
    variant: value_objects.PredictorVariant,
    command: run_ablation.AblationCommand,
    *,
    tau: float,
    code: float,
) -> ports.IPredictor:
    """Builds the physics + LSTM (with context) or Transformer of E9b."""
    del command  # Part of the factory signature; not needed here.
    if variant == value_objects.PredictorVariant.TRANSFORMER:
        return registry.build_model("transformer_large")
    return make_context_predictor(code, 0.0, tau)


def run_e9a(simulator, groups, n_epochs: int,
            verbose: bool = True) -> list[dict[str, object]]:
    """E9a: context-aware physics + LSTM trained on D+E, scored per scenario.

    Training reads the 4-class codes from the dataset (D=2, E=3); at
    evaluation `eval_interaction_codes` sets the code of each group.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per evaluation scenario, with the keys of `E9A_COLUMNS`.
    """
    results = run_cross_system.CrossSystemUseCase(
        simulator, store=scenario_lib.trajectory_store(),
        predictor_factory=functools.partial(make_context_predictor, 2.5),
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
        eval_interaction_codes={"D": 2.0, "E": 3.0},
    ))
    rows = [result.as_dict() for result in results]
    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for row in rows:
        print(f"{row['name']:>20} {row['r2']:>8.4f} {row['mae']:>8.4f} "
              f"{row['auroc']:>8.4f} {row['n_samples']:>6}")
    return rows


def run_e9b(simulator, groups, n_epochs: int,
            verbose: bool = True) -> list[dict[str, object]]:
    """E9b: physics + LSTM with context codes vs the Transformer.

    Args:
        simulator: The simulator.
        groups: Scenario label -> configurations.
        n_epochs: Training epochs.
        verbose: Whether to print training progress.

    Returns:
        One row per scenario and variant, with the keys of `E9B_COLUMNS`.
    """
    variants = value_objects.PredictorVariant
    rows = []
    for label, code in (("D", 2.0), ("E", 3.0)):
        tau = 0.5 if label == "D" else 0.0
        print(f"\n{'=' * 60}\nE9b — Scenario {label}  "
              f"(context code={code})\n{'=' * 60}")
        results = run_ablation.AblationStudyUseCase(
            simulator, store=scenario_lib.trajectory_store(),
            predictor_factory=functools.partial(
                _ablation_predictor, tau=tau, code=code),
        ).execute(run_ablation.AblationCommand(
            configs=groups[label],
            variants=[variants.PHYSICS_LSTM, variants.TRANSFORMER],
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
            rows.append(
                {**result.as_dict(), "scenario": label, "context_code": code})

    print(f"\n{'Scenario':<8} {'Code':>6} {'Variant':<14} {'R²':>8} "
          f"{'AUROC':>8}")
    print("-" * 50)
    for row in rows:
        print(f"{row['scenario']:<8} {row['context_code']:>6.1f} "
              f"{row['variant']:<14} {row['r2']:>8.4f} {row['auroc']:>8.4f}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E9a and E9b and writes their CSVs."""
    args = common.parse_args(__doc__, argv=argv, seeds=None, out=False)
    n_per, epochs = (50, 10) if args.fast else (500, 150)
    config = {"n_per": n_per, "n_epochs": epochs, "window": WINDOW_2Q,
              "min_gap": MIN_GAP}
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'=' * 70}\nE9a — Context cross-system | n_per={n_per} | "
          f"epochs={epochs}\n{'=' * 70}")
    rows = run_e9a(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e9a_cross_system", args),
        columns=E9A_COLUMNS, seeds=[42], config=config)

    print(f"\n{'=' * 70}\nE9b — Context ablation (D, E) | n_per={n_per} | "
          f"epochs={epochs}\n{'=' * 70}")
    rows = run_e9b(simulator, groups, epochs)
    common.write_results(
        rows, common.results_path("e9b_ablation", args),
        columns=E9B_COLUMNS, seeds=[42], config=config)
    print("\nE9 completed. Results in experiments/results/e9*.csv")


if __name__ == "__main__":
    main()
