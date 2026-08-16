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
experiments/results/e9a_cross_system.csv
experiments/results/e9b_ablation.csv

Usage
-----
    python -m experiments.e9_context_injection
    python -m experiments.e9_context_injection --fast
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"
WINDOW_2Q   = 30
MIN_GAP     = 15


def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def _make_context_predictor(interaction_code: float = 0.5, dissipator_code: float = 0.0):
    """Build the context-aware physics_lstm."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor
    return LSTMPredictor(
        hidden_size=128, num_layers=2, dropout=0.3,
        use_physics_prior=True,
        adaptive_prior_r2_threshold=0.5,
        risk_pos_weight=0.0, augment_sigma=0.02,
        inference_interaction_code=interaction_code,
        inference_dissipator_code=dissipator_code,
    )


# ---------------------------------------------------------------------------
# E9a — cross-system with context codes
# ---------------------------------------------------------------------------

def run_e9a(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """Cross-system: physics_lstm+context trained on D+E, eval per scenario."""

    cmd = CrossSystemCommand(
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
    )

    class E9CrossSystem(CrossSystemUseCase):
        def _build_predictor(self):
            # Training reads 4-class codes from the dataset (D=2, E=3).
            # Eval overrides inference_interaction_code per group via
            # CrossSystemCommand.eval_interaction_codes.
            return _make_context_predictor(interaction_code=2.5)

    results = E9CrossSystem(simulator, store=trajectory_store()).execute(cmd)
    rows = [r.as_dict() for r in results]
    _save_csv(rows, RESULTS_DIR / "e9a_cross_system.csv",
              ["name", "variant", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])

    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for r in results:
        print(f"{r.spec.name:>20} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f} {r.n_samples:>6}")
    return rows


# ---------------------------------------------------------------------------
# E9b — ablation: context vs no-context on single-scenario
# ---------------------------------------------------------------------------

def run_e9b(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """Ablation: compare physics_lstm with and without context codes."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    all_rows = []
    for scenario_label, int_code in [("D", 2.0), ("E", 3.0)]:
        tau = 0.5 if scenario_label == "D" else 0.0
        print(f"\n{'='*60}\nE9b — Scenario {scenario_label}  (context code={int_code})\n{'='*60}")

        cmd = AblationCommand(
            configs=groups[scenario_label],
            variants=[PredictorVariant.PHYSICS_LSTM, PredictorVariant.TRANSFORMER],
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
        )

        _ic = int_code
        _tau = tau

        class E9Ablation(AblationStudyUseCase):
            def _build_predictor(self, variant, command):
                from src.infrastructure.ml.transformer_predictor import TransformerPredictor
                if variant == PredictorVariant.TRANSFORMER:
                    return TransformerPredictor(
                        d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1,
                    )
                return LSTMPredictor(
                    hidden_size=128, num_layers=2, dropout=0.3,
                    use_physics_prior=True,
                    adaptive_prior_r2_threshold=_tau,
                    risk_pos_weight=0.0, augment_sigma=0.02,
                    inference_interaction_code=_ic,
                    inference_dissipator_code=0.0,
                )

        results = E9Ablation(simulator, store=trajectory_store()).execute(cmd)
        for r in results:
            row = r.as_dict()
            row["scenario"] = scenario_label
            row["context_code"] = int_code
            all_rows.append(row)

    _save_csv(all_rows, RESULTS_DIR / "e9b_ablation.csv",
              ["scenario", "context_code", "name", "variant",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])

    print(f"\n{'Scenario':<8} {'Code':>6} {'Variant':<14} {'R²':>8} {'AUROC':>8}")
    print("-" * 50)
    for r in all_rows:
        print(f"{r['scenario']:<8} {r['context_code']:>6.1f} {r['variant']:<14} "
              f"{r['r2']:>8.4f} {r['auroc']:>8.4f}")
    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_per  = 50  if fast else 500
    epochs = 10  if fast else 150

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'='*70}")
    print(f"E9a — Context cross-system | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    run_e9a(simulator, groups, n_epochs=epochs)

    print(f"\n{'='*70}")
    print(f"E9b — Context ablation (D, E) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    run_e9b(simulator, groups, n_epochs=epochs)

    print("\nE9 completed. Results in experiments/results/e9_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
