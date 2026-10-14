"""Experiment E2 — Minimum observation window sweep.

Goal
----
Find how short the observation window can be for a usable early-warning
alarm. Success is risk AUROC, not regression R².

Claim
-----
C3: f = t_obs/T₂ = 0.15 is enough for AUROC ≥ 0.92. R² plateaus near 0.73
    — do not claim R² ≥ 0.95.

Scenarios
---------
Default: B+C (1q σ₋ / σ_z time-dep γ). Optional: D / E via --scenarios.

Metrics
-------
AUROC ≥ 0.92 at f = 0.15 (E2: 0.947). Report MAE and R² vs f as well.
Sweep f ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.

Output
------
experiments/results/e2_window_sweep[_<tag>][_fast].csv
columns: name, variant, window_fraction, noise_sigma, mae, rmse, r2, mape,
auroc, n_samples (Paper 1 uses --tag p1).

Usage
-----
    python -m experiments.e2_window_sweep --fast
    python -m experiments.e2_window_sweep --tag p1
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import run_window_sweep
from src.data import scenarios as scenario_lib
from src.simulation import simulator as simulator_lib

COLUMNS = ("name", "variant", "window_fraction", "noise_sigma", "mae",
           "rmse", "r2", "mape", "auroc", "n_samples")
FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
AUROC_TARGET = 0.92


def run(
    scenarios: Sequence[str] = ("B", "C"),
    n_per: int = 120,
    *,
    seed: int = 42,
    n_epochs: int = 100,
) -> list[dict[str, object]]:
    """Trains one model per window fraction.

    Args:
        scenarios: Scenario labels pooled into one dataset.
        n_per: Trajectories per scenario.
        seed: Seed of the configurations, dataset and training.
        n_epochs: Training epochs.

    Returns:
        One row per window fraction with the keys of `COLUMNS`.
    """
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=seed)
    configs = [cfg for label in scenarios for cfg in groups[label]]
    results = run_window_sweep.WindowSweepUseCase(
        simulator_lib.QuTipSimulator(), store=scenario_lib.trajectory_store()
    ).execute(run_window_sweep.WindowSweepCommand(
        configs=configs,
        fractions=list(FRACTIONS),
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


def _print_summary(rows: Sequence[dict[str, object]]) -> None:
    print(f"\n{'f':>6} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 36)
    for row in rows:
        star = " *" if row["auroc"] >= AUROC_TARGET else ""
        print(f"{row['window_fraction']:>6.2f} {row['r2']:>8.4f} "
              f"{row['mae']:>8.4f} {row['auroc']:>8.4f}{star}")
    above = [row for row in rows if row["auroc"] >= AUROC_TARGET]
    if above:
        best = min(row["window_fraction"] for row in above)
        print(f"\nMinimum window fraction f* (AUROC≥0.92): {best:.2f}")
    else:
        print("\nAUROC≥0.92 not achieved at any window fraction.")


def _add_flags(parser) -> None:
    parser.add_argument("--scenarios", nargs="+", default=["B", "C"])


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E2 and writes its CSV."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, seeds=None)
    n_per, n_epochs = (30, 10) if args.fast else (120, 100)
    rows = run(args.scenarios, n_per, n_epochs=n_epochs)
    path = common.results_path("e2_window_sweep", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"scenarios": args.scenarios, "n_per": n_per,
                "fractions": FRACTIONS, "n_epochs": n_epochs})
    print(f"\nResults saved to {path}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
