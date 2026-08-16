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
Default: B, C (time-dep γ). Optional: D, E via --scenarios.

Metrics
-------
R², MAE, AUROC at σ ∈ {0, 0.01, 0.02, 0.05, 0.10}.
Train on clean trajectories; noise is added at inference only.

Output
------
experiments/results/e3_noise_sweep_{suffix}.csv
columns: scenario, noise_sigma, r2, mae, auroc

Usage
-----
    python -m experiments.e3_noise_sweep
    python -m experiments.e3_noise_sweep --scenarios D E --suffix p2
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_noise_sweep import NoiseSweepCommand, NoiseSweepUseCase
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def main(scenarios: list | None = None, suffix: str = "") -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=100, seed=42)
    # Default: B+C+D (mixed 1-qubit/2-qubit time-dep)
    if scenarios is None:
        scenarios = ["B", "C", "D"]
    configs = [cfg for s in scenarios for cfg in groups[s]]

    cmd = NoiseSweepCommand(
        configs=configs,
        sigmas=[0.0, 0.01, 0.02, 0.05, 0.10],
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

    results = NoiseSweepUseCase(simulator, store=trajectory_store()).execute(cmd)

    fname = f"e3_noise_sweep{suffix}.csv" if suffix else "e3_noise_sweep.csv"
    out_path = results_dir / fname
    fieldnames = ["name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([r.as_dict() for r in results])

    print(f"\nResults saved to {out_path}")

    print(f"\n{'σ_noise':>9} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 38)
    for r in results:
        flag = " ✓" if r.risk_auroc >= 0.90 else " ✗"
        print(f"{r.spec.noise.sigma:>9.3f} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f}{flag}")


if __name__ == "__main__":
    main()
