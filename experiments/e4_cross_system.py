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
Paper 2: --train_scenarios D E --eval_scenarios D E.

Metrics
-------
R², MAE, AUROC, n_samples per eval scenario.

Output
------
experiments/results/e4_cross_system_{suffix}.csv
columns: train_set, eval_scenario, model, r2, mae, auroc, n_samples

Usage
-----
    python -m experiments.e4_cross_system
    python -m experiments.e4_cross_system --train_scenarios D E --eval_scenarios D E --suffix p2
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def main(train_scenarios: list | None = None, eval_scenarios: list | None = None, suffix: str = "") -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=100, seed=42)

    # Defaults: train on all, eval on all individual scenarios
    if train_scenarios is None:
        train_scenarios = ["A", "B", "C", "D", "E"]
    if eval_scenarios is None:
        eval_scenarios = ["A", "B", "C", "D", "E"]

    train_configs = [cfg for s in train_scenarios for cfg in groups[s]]
    eval_groups = {s: groups[s] for s in eval_scenarios}

    # E4: train on selected scenarios, evaluate per scenario
    cmd = CrossSystemCommand(
        train_configs=train_configs,
        eval_groups=eval_groups,
        window_length=50,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=100,
        batch_size=32,
        lr=1e-3,
        regression_loss="huber",
        seed=42,
        verbose=True,
    )

    results = CrossSystemUseCase(simulator, store=trajectory_store()).execute(cmd)

    fname = f"e4_cross_system{suffix}.csv" if suffix else "e4_cross_system.csv"
    out_path = results_dir / fname
    fieldnames = ["name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([r.as_dict() for r in results])

    print(f"\nResults saved to {out_path}")

    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for r in results:
        print(
            f"{r.spec.name:>20} {r.r2:>8.4f} {r.mae:>8.4f} "
            f"{r.risk_auroc:>8.4f} {r.n_samples:>6}"
        )


if __name__ == "__main__":
    main()
