"""Experiment E4 — Cross-system generalisation.

Goal
----
Demonstrate that a single model trained on a mixture of all system types
(1-qubit + 2-qubit, XXZ + TFIM, σ₋ + σ_z) generalises to each subsystem
without per-system retraining.

Claim
-----
C5: A single model trained on diverse scenarios generalises across all
    regimes without retraining.

Scenarios
---------
Default: train A+B+C+D+E, eval each separately.
Paper 1: --train-scenarios A B C --eval-scenarios A B C --tag p1.
Paper 2: --train-scenarios D E --eval-scenarios D E.

Metrics
-------
R², MAE, AUROC, n_samples per eval scenario.

Output
------
experiments/results/e4_cross_system[_<tag>][_fast].csv
columns: name, variant, window_fraction, noise_sigma, mae, rmse, r2, mape,
auroc, n_samples

Usage
-----
    python -m experiments.e4_cross_system --fast
    python -m experiments.e4_cross_system --train-scenarios D E \\
        --eval-scenarios D E --tag p2
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import run_cross_system
from src.data import scenarios as scenario_lib
from src.simulation import simulator as simulator_lib

COLUMNS = ("name", "variant", "window_fraction", "noise_sigma", "mae",
           "rmse", "r2", "mape", "auroc", "n_samples")
ALL_SCENARIOS = ("A", "B", "C", "D", "E")


def run(
    train_scenarios: Sequence[str] = ALL_SCENARIOS,
    eval_scenarios: Sequence[str] = ALL_SCENARIOS,
    n_per: int = 100,
    *,
    seed: int = 42,
    n_epochs: int = 100,
) -> list[dict[str, object]]:
    """Trains one model on a mixture and scores it per scenario.

    Args:
        train_scenarios: Scenario labels of the training mixture.
        eval_scenarios: Scenario labels scored separately.
        n_per: Trajectories per scenario.
        seed: Seed of the configurations, datasets and training.
        n_epochs: Training epochs.

    Returns:
        One row per evaluation scenario with the keys of `COLUMNS`.
    """
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=seed)
    results = run_cross_system.CrossSystemUseCase(
        simulator_lib.QuTipSimulator(), store=scenario_lib.trajectory_store()
    ).execute(run_cross_system.CrossSystemCommand(
        train_configs=[cfg for label in train_scenarios
                       for cfg in groups[label]],
        eval_groups={label: groups[label] for label in eval_scenarios},
        window_length=50,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=32,
        lr=1e-3,
        regression_loss="huber",
        seed=seed,
        verbose=True,
    ))
    return [result.as_dict() for result in results]


def _add_flags(parser) -> None:
    parser.add_argument("--train-scenarios", nargs="+",
                        default=list(ALL_SCENARIOS))
    parser.add_argument("--eval-scenarios", nargs="+",
                        default=list(ALL_SCENARIOS))


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E4 and writes its CSV."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, seeds=None)
    n_per, n_epochs = (30, 10) if args.fast else (100, 100)
    rows = run(args.train_scenarios, args.eval_scenarios, n_per,
               n_epochs=n_epochs)
    path = common.results_path("e4_cross_system", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"train_scenarios": args.train_scenarios,
                "eval_scenarios": args.eval_scenarios, "n_per": n_per,
                "n_epochs": n_epochs})
    print(f"\nResults saved to {path}")
    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for row in rows:
        print(f"{row['name']:>20} {row['r2']:>8.4f} {row['mae']:>8.4f} "
              f"{row['auroc']:>8.4f} {row['n_samples']:>6}")


if __name__ == "__main__":
    main()
