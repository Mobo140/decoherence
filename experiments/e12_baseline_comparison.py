"""Experiment E12 — Classical baseline comparison.

Goal
----
Check that physics_lstm beats a non-ML stretched-exponential curve_fit
baseline on time-dependent γ (claim C2).  Without this, C2 is not
falsifiable: a reviewer can ask whether ordinary curve_fit already
explains the R².

Claim
-----
C2: physics+LSTM residual outperforms physics-only AND stretched-exp
    for time-dependent γ(t).

Scenarios
---------
B (1q σ₋ time-dep), C (1q σ_z time-dep), D (2q XXZ), E (2q TFIM).

Metrics
-------
R², MAE. physics_only = OLS log C; stretched_exp = curve_fit C0·exp(-(t/τ)^β).

Output
------
experiments/results/e12_baseline_comparison[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e12_baseline_comparison
    python -m experiments.e12_baseline_comparison --fast
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import run_ablation
from src.data import scenarios as scenario_lib
from src.domain import value_objects
from src.simulation import simulator as simulator_lib

COLUMNS = (
    "scenario", "name", "variant", "window_fraction", "noise_sigma", "mae",
    "rmse", "r2", "mape", "auroc", "n_samples")
VARIANTS = (
    value_objects.PredictorVariant.PHYSICS_ONLY,
    value_objects.PredictorVariant.STRETCHED_EXP,
    value_objects.PredictorVariant.PHYSICS_LSTM,
)
SCENARIOS = ("B", "C", "D", "E")


def run(n_per: int, n_epochs: int, *,
        seed: int = 42) -> list[dict[str, object]]:
    """Scores the physics-only, stretched-exp and physics + LSTM models.

    Args:
        n_per: Trajectories per scenario.
        n_epochs: Training epochs.
        seed: Seed of the configurations, datasets and training.

    Returns:
        One row per scenario and variant, with the keys of `COLUMNS`.
    """
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=seed)
    use_case = run_ablation.AblationStudyUseCase(
        simulator_lib.QuTipSimulator(),
        store=scenario_lib.trajectory_store())
    rows = []
    for label in SCENARIOS:
        print(f"\n{'=' * 60}\nE12 — Scenario {label}  n_per={n_per}  "
              f"epochs={n_epochs}\n{'=' * 60}")
        results = use_case.execute(run_ablation.AblationCommand(
            configs=groups[label],
            variants=list(VARIANTS),
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


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E12 and writes its CSV."""
    args = common.parse_args(__doc__, argv=argv, seeds=None)
    n_per, n_epochs = (20, 15) if args.fast else (100, 80)
    rows = run(n_per, n_epochs)
    path = common.results_path("e12_baseline_comparison", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"n_per": n_per, "n_epochs": n_epochs})
    print(f"\nResults saved to {path}")
    print(f"\n{'Scenario':<10} {'Variant':<16} {'R²':>8} {'MAE':>8} "
          f"{'AUROC':>8}")
    print("-" * 54)
    for row in rows:
        print(f"{row['scenario']:<10} {row['variant']:<16} "
              f"{row['r2']:>8.4f} {row['mae']:>8.4f} {row['auroc']:>8.4f}")


if __name__ == "__main__":
    main()
