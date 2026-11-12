"""Experiment E3 — Measurement noise robustness.

Goal
----
Show that the model degrades gracefully under i.i.d. Gaussian measurement
noise, bridging the gap between ideal simulation and real quantum hardware.

Claim
-----
C4: Adding Gaussian noise σ_noise to observables degrades MAE sub-linearly;
    AUROC remains ≥0.90 at realistic noise levels (σ≤0.05).

Scenarios
---------
Default: B, C, D (time-dep γ). Choose with --scenarios.

Metrics
-------
R², MAE, AUROC at σ ∈ {0, 0.01, 0.02, 0.05, 0.10}.
Train on clean trajectories; noise is added at inference only.

Output
------
experiments/results/e3_noise_sweep[_<tag>][_fast].csv
columns: name, variant, window_fraction, noise_sigma, mae, rmse, r2, mape,
auroc, n_samples (Paper 1 uses --scenarios B C --tag p1).

Usage
-----
    python -m experiments.e3_noise_sweep --fast
    python -m experiments.e3_noise_sweep --scenarios B C --tag p1
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from src.application import run_noise_sweep
from src.data import scenarios as scenario_lib
from src.simulation import simulator as simulator_lib

COLUMNS = ("name", "variant", "window_fraction", "noise_sigma", "mae",
           "rmse", "r2", "mape", "auroc", "n_samples")
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)
AUROC_TARGET = 0.90


def run(
    scenarios: Sequence[str] = ("B", "C", "D"),
    n_per: int = 100,
    *,
    seed: int = 42,
    n_epochs: int = 100,
) -> list[dict[str, object]]:
    """Trains on clean data and scores each noise level.

    Args:
        scenarios: Scenario labels pooled into one dataset.
        n_per: Trajectories per scenario.
        seed: Seed of the configurations, dataset, training and noise.
        n_epochs: Training epochs.

    Returns:
        One row per sigma with the keys of `COLUMNS`.
    """
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=seed)
    configs = [cfg for label in scenarios for cfg in groups[label]]
    results = run_noise_sweep.NoiseSweepUseCase(
        simulator_lib.QuTipSimulator(), store=scenario_lib.trajectory_store()
    ).execute(run_noise_sweep.NoiseSweepCommand(
        configs=configs,
        sigmas=list(SIGMAS),
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
    parser.add_argument("--scenarios", nargs="+", default=["B", "C", "D"])


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E3 and writes its CSV."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, seeds=None)
    n_per, n_epochs = (30, 10) if args.fast else (100, 100)
    rows = run(args.scenarios, n_per, n_epochs=n_epochs)
    path = common.results_path("e3_noise_sweep", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"scenarios": args.scenarios, "n_per": n_per,
                "sigmas": SIGMAS, "n_epochs": n_epochs})
    print(f"\nResults saved to {path}")
    print(f"\n{'σ_noise':>9} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 38)
    for row in rows:
        flag = " ✓" if row["auroc"] >= AUROC_TARGET else " ✗"
        print(f"{row['noise_sigma']:>9.3f} {row['r2']:>8.4f} "
              f"{row['mae']:>8.4f} {row['auroc']:>8.4f}{flag}")


if __name__ == "__main__":
    main()
