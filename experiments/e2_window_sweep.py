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
Default: B (1q σ₋ time-dep γ). Optional: C / D / E via --scenarios.

Metrics
-------
AUROC ≥ 0.92 at f = 0.15 (E2: 0.947). Report MAE and R² vs f as well.
Sweep f ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.

Output
------
experiments/results/e2_window_sweep_{suffix}.csv
columns: scenario, window_frac, r2, mae, auroc

Usage
-----
    python -m experiments.e2_window_sweep
    python -m experiments.e2_window_sweep --scenarios B C --suffix p1
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_window_sweep import WindowSweepCommand, WindowSweepUseCase
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def main(scenarios: list | None = None, suffix: str = "") -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=120, seed=42)

    # Default: B+C (1-qubit time-dependent γ)
    if scenarios is None:
        scenarios = ["B", "C"]
    configs = [cfg for s in scenarios for cfg in groups[s]]

    cmd = WindowSweepCommand(
        configs=configs,
        fractions=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=100,
        batch_size=32,
        lr=1e-3,
        regression_loss="huber",
        seed=42,
        verbose=True,
    )

    results = WindowSweepUseCase(simulator, store=trajectory_store()).execute(cmd)

    fname = f"e2_window_sweep{suffix}.csv" if suffix else "e2_window_sweep.csv"
    out_path = results_dir / fname
    fieldnames = ["name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([r.as_dict() for r in results])

    print(f"\nResults saved to {out_path}")

    print(f"\n{'f':>6} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 36)
    for r in results:
        star = " *" if r.risk_auroc >= 0.92 else ""
        print(f"{r.spec.window_fraction:>6.2f} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f}{star}")

    above = [r for r in results if r.risk_auroc >= 0.92]
    if above:
        f_star = min(r.spec.window_fraction for r in above)
        print(f"\nMinimum window fraction f* (AUROC≥0.92): {f_star:.2f}")
    else:
        print("\nAUROC≥0.92 not achieved at any window fraction.")


if __name__ == "__main__":
    main()
