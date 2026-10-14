"""Experiment E1 — Ablation study: architecture components.

Goal
----
Verify that the physics-informed residual architecture outperforms both
physics-only and pure-ML baselines when γ(t) is unknown and time-dependent.

Claim
-----
C1: OLS-slope formula achieves R²≥0.999 and MAE≤0.05 for constant-γ + σ₋.
    (Validates that the physics prior is exact in the tractable limit.)
C2: physics+LSTM residual systematically outperforms physics-only and lstm-only
    for time-dependent γ(t) across all parameterised γ shapes.
    (Core technical contribution of Paper 1.)

Models compared
---------------
  - physics_only      (OLS slope formula, no ML)
  - lstm_only         (BiLSTM without physics prior)
  - physics_lstm      (residual: physics + LSTM)  ← our main contribution
  - transformer       (pure Transformer, 2505.06928-style baseline)
  - stretched_exp     (curve-fit baseline, no ML)

Scenarios
---------
Default: A (1q σ₋ const γ), B (1q σ₋ time-dep γ), C (1q σ_z time-dep γ),
D (2q XXZ), E (2q TFIM); choose with --scenarios.

Metrics
-------
R², MAE, RMSE, MAPE, AUROC (risk head)

Output
------
experiments/results/e1_ablation[_<tag>][_fast].csv
(Paper 1 uses --scenarios A B C --n-per 100 --tag p1.)

Usage
-----
    python -m experiments.e1_ablation --fast
    python -m experiments.e1_ablation --scenarios A B C --n-per 100 --tag p1
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import run_ablation
from src.data import scenarios as scenario_lib
from src.domain import value_objects
from src.simulation import simulator as simulator_lib

COLUMNS = ("scenario", "name", "variant", "window_fraction", "noise_sigma",
           "mae", "rmse", "r2", "mape", "auroc", "n_samples")
ALL_SCENARIOS = ("A", "B", "C", "D", "E")
DEFAULT_N_PER = 80


def run(
    scenarios: Sequence[str] = ALL_SCENARIOS,
    n_per: int = DEFAULT_N_PER,
    *,
    seed: int = 42,
    n_epochs: int = 100,
) -> list[dict[str, object]]:
    """Runs the ablation on each scenario separately.

    Args:
        scenarios: Scenario labels to run.
        n_per: Trajectories per scenario.
        seed: Seed of the configurations, dataset and training.
        n_epochs: Training epochs.

    Returns:
        One row per scenario and variant, with the keys of `COLUMNS`.
    """
    simulator = simulator_lib.QuTipSimulator()
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=seed)
    use_case = run_ablation.AblationStudyUseCase(
        simulator, store=scenario_lib.trajectory_store())
    rows = []
    for label in scenarios:
        print(f"\n{'=' * 60}\nScenario {label}\n{'=' * 60}")
        results = use_case.execute(run_ablation.AblationCommand(
            configs=groups[label],
            variants=list(value_objects.PredictorVariant),
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=n_epochs,
            batch_size=32,
            lr=1e-3,
            regression_loss="huber",
            seed=seed,
            verbose=True,
        ))
        for result in results:
            rows.append({**result.as_dict(), "scenario": label})
    return rows


def _print_summary(rows: Sequence[dict[str, object]]) -> None:
    print(f"\n{'Scenario':<10} {'Variant':<16} {'R²':>8} {'MAE':>8} "
          f"{'AUROC':>8}")
    print("-" * 54)
    for row in rows:
        print(f"{row['scenario']:<10} {row['variant']:<16} "
              f"{row['r2']:>8.4f} {row['mae']:>8.4f} {row['auroc']:>8.4f}")


def _add_flags(parser) -> None:
    parser.add_argument("--scenarios", nargs="+", default=list(ALL_SCENARIOS))
    parser.add_argument("--n-per", type=int, default=DEFAULT_N_PER,
                        help="trajectories per scenario")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E1 and writes its CSV."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, seeds=None)
    n_per, n_epochs = (20, 10) if args.fast else (args.n_per, 100)
    rows = run(args.scenarios, n_per, n_epochs=n_epochs)
    path = common.results_path("e1_ablation", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"scenarios": args.scenarios, "n_per": n_per,
                "n_epochs": n_epochs})
    print(f"\nResults saved to {path}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
