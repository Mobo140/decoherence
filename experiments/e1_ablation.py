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

Scenarios
---------
Default: A (1q σ₋ const γ), B (1q σ₋ time-dep γ), C (1q σ_z time-dep γ)
Optional: D (2q XXZ), E (2q TFIM) via --scenarios D E

Metrics
-------
R², MAE, RMSE, MAPE, AUROC (risk head)

Output
------
experiments/results/e1_ablation_{suffix}.csv

Usage
-----
    python -m experiments.e1_ablation
    python -m experiments.e1_ablation --scenarios D E --suffix p2
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def main(scenarios: list | None = None, n_per: int | None = None, suffix: str = "",
         *, seed: int = 42, out_path=None) -> list:
    """Defaults reproduce the published E1 run.

    The keyword-only arguments exist so R7 can replicate it across seeds
    without duplicating the configuration; see
    experiments/r7_multiseed_1qubit.py.
    """
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    simulator = QuTipSimulator()
    if n_per is None:
        n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    groups = build_configs(n_per_scenario=n_per, seed=seed)

    if scenarios is None:
        scenarios = ["A", "B", "C", "D", "E"]

    all_rows = []

    # --- Run on each scenario separately ---
    for scenario_label in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario {scenario_label}")
        print(f"{'='*60}")

        cmd = AblationCommand(
            configs=groups[scenario_label],
            variants=list(PredictorVariant),
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=100,
            batch_size=32,
            lr=1e-3,
            regression_loss="huber",
            seed=seed,
            verbose=True,
        )

        results = AblationStudyUseCase(simulator, store=trajectory_store()).execute(cmd)

        for r in results:
            row = r.as_dict()
            row["scenario"] = scenario_label
            all_rows.append(row)

    # --- Save CSV ---
    fname = f"e1_ablation{suffix}.csv" if suffix else "e1_ablation.csv"
    out_path = out_path or results_dir / fname
    fieldnames = ["scenario", "name", "variant", "window_fraction", "noise_sigma",
                  "mae", "rmse", "r2", "mape", "auroc", "n_samples"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {out_path}")
    _print_summary(all_rows)
    return all_rows


def _print_summary(rows: list) -> None:
    print(f"\n{'Scenario':<10} {'Variant':<16} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 54)
    for r in rows:
        print(f"{r['scenario']:<10} {r['variant']:<16} {r['r2']:>8.4f} {r['mae']:>8.4f} {r['auroc']:>8.4f}")


if __name__ == "__main__":
    main()
