"""Experiment E8 — Improved two-qubit training with E7 findings applied.

Goal
----
Apply E7b τ and four training fixes to physics_lstm; keep the best
valid TFIM cross-system number (E8c R²=0.575).

Claim
-----
C2/C5: per-scenario τ (D=0.5, E=0.0) + weighted BCE + noise aug
+ window=30 + min_gap=15 beat E6c on D and E.

Scenarios
---------
D (2q XXZ), E (2q TFIM).
E8a ablation, E8b noise, E8c cross-system (train D+E).

Metrics
-------
R², MAE, AUROC. Paper 2 baseline: E8c D=0.740, E=0.575.

Output
------
experiments/results/e8a_ablation.csv
experiments/results/e8b_noise_sweep.csv
experiments/results/e8c_cross_system.csv

Usage
-----
    python -m experiments.e8_improved_2qubit
    python -m experiments.e8_improved_2qubit --fast
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.application.run_noise_sweep import NoiseSweepCommand, NoiseSweepUseCase
from src.application.run_cross_system import CrossSystemCommand, CrossSystemUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


RESULTS_DIR = Path(__file__).parent / "results"
WINDOW_2Q = 30       # wider window for 2-qubit physics slope stability
MIN_GAP   = 15       # min window gap (half of WINDOW_2Q)


def _save_csv(rows: list, path: Path, fieldnames: list) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_rerun{path.suffix}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Per-scenario τ mapping (from E7b results)
# ---------------------------------------------------------------------------

TAU_PER_SCENARIO = {"D": 0.5, "E": 0.0}


# ---------------------------------------------------------------------------
# E8a — ablation
# ---------------------------------------------------------------------------

def run_e8a(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """Ablation: physics_lstm(improved) vs Transformer on D and E."""
    all_rows = []

    for scenario_label in ["D", "E"]:
        tau = TAU_PER_SCENARIO[scenario_label]
        print(f"\n{'='*60}\nE8a — Scenario {scenario_label}  (τ={tau})\n{'='*60}")

        cmd = AblationCommand(
            configs=groups[scenario_label],
            variants=list(PredictorVariant),
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

        _tau = tau  # capture for closure

        class E8Ablation(AblationStudyUseCase):
            def _build_predictor(self, variant, command):
                from src.infrastructure.ml.lstm_predictor import LSTMPredictor
                from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
                from src.infrastructure.ml.transformer_predictor import TransformerPredictor
                if variant == PredictorVariant.PHYSICS_ONLY:
                    return PhysicsOnlyPredictor(t_max=max(c.t_max for c in command.configs))
                if variant == PredictorVariant.LSTM_ONLY:
                    return LSTMPredictor(
                        hidden_size=128, num_layers=2, dropout=0.3,
                        use_physics_prior=False,
                        risk_pos_weight=0.0, augment_sigma=0.02,
                    )
                if variant == PredictorVariant.PHYSICS_LSTM:
                    return LSTMPredictor(
                        hidden_size=128, num_layers=2, dropout=0.3,
                        use_physics_prior=True,
                        adaptive_prior_r2_threshold=_tau,
                        risk_pos_weight=0.0, augment_sigma=0.02,
                    )
                if variant == PredictorVariant.TRANSFORMER:
                    return TransformerPredictor(
                        d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1,
                    )
                raise ValueError(variant)

        results = E8Ablation(simulator, store=trajectory_store()).execute(cmd)
        for r in results:
            row = r.as_dict()
            row["scenario"] = scenario_label
            row["tau"] = tau
            all_rows.append(row)

    _save_csv(all_rows, RESULTS_DIR / "e8a_ablation.csv",
              ["scenario", "tau", "name", "variant", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])

    print(f"\n{'Scenario':<8} {'τ':>5} {'Variant':<16} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 58)
    for r in all_rows:
        print(f"{r['scenario']:<8} {r['tau']:>5.1f} {r['variant']:<16} "
              f"{r['r2']:>8.4f} {r['mae']:>8.4f} {r['auroc']:>8.4f}")
    return all_rows


# ---------------------------------------------------------------------------
# E8b — noise sweep
# ---------------------------------------------------------------------------

def run_e8b(simulator, groups, n_epochs: int, verbose: bool = True) -> list:
    """Noise robustness: improved physics_lstm on D+E mixture."""
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    configs = groups["D"] + groups["E"]
    cmd = NoiseSweepCommand(
        configs=configs,
        sigmas=[0.0, 0.01, 0.02, 0.05, 0.10],
        window_length=WINDOW_2Q,
        horizon=1.0,
        samples_per_trajectory=5,
        n_epochs=n_epochs,
        batch_size=64,
        lr=5e-4,
        regression_loss="huber",
        seed=seed,
        verbose=verbose,
        min_window_gap=MIN_GAP,
    )

    class E8NoiseSweep(NoiseSweepUseCase):
        def _build_predictor(self):
            return LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=True,
                adaptive_prior_r2_threshold=0.5,  # compromise between D(0.5) and E(0.0)
                risk_pos_weight=0.0, augment_sigma=0.02,
            )

    results = E8NoiseSweep(simulator).execute(cmd)
    rows = [r.as_dict() for r in results]
    _save_csv(rows, RESULTS_DIR / "e8b_noise_sweep.csv",
              ["name", "variant", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])

    print(f"\n{'σ_noise':>9} {'R²':>8} {'MAE':>8} {'AUROC':>8}")
    print("-" * 38)
    for r in results:
        flag = " ✓" if r.risk_auroc >= 0.90 else " ✗"
        print(f"{r.spec.noise.sigma:>9.3f} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f}{flag}")
    return rows


# ---------------------------------------------------------------------------
# E8c — cross-system
# ---------------------------------------------------------------------------

def run_e8c(simulator, groups, n_epochs: int, verbose: bool = True,
            *, seed: int = 42, out_path=None) -> list:
    """Cross-system: improved physics_lstm trained on D+E, eval per scenario.

    Defaults reproduce the published run; see _noise_sweep in
    e6_2qubit_improved.py for why the keyword-only arguments exist.
    """
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

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
    )

    class E8CrossSystem(CrossSystemUseCase):
        def _build_predictor(self):
            return LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=True,
                adaptive_prior_r2_threshold=0.5,
                risk_pos_weight=0.0, augment_sigma=0.02,
            )

    results = E8CrossSystem(simulator, store=trajectory_store()).execute(cmd)
    rows = [r.as_dict() for r in results]
    _save_csv(rows, out_path or RESULTS_DIR / "e8c_cross_system.csv",
              ["name", "variant", "window_fraction", "noise_sigma",
               "mae", "rmse", "r2", "mape", "auroc", "n_samples"])

    print(f"\n{'Scenario':>20} {'R²':>8} {'MAE':>8} {'AUROC':>8} {'N':>6}")
    print("-" * 56)
    for r in results:
        print(f"{r.spec.name:>20} {r.r2:>8.4f} {r.mae:>8.4f} {r.risk_auroc:>8.4f} {r.n_samples:>6}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False) -> None:
    n_per  = 50  if fast else 500
    epochs = 10  if fast else 150

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=n_per, seed=42)

    print(f"\n{'='*70}")
    print(f"E8a — Ablation (D+E) | n_per={n_per} | epochs={epochs} | window={WINDOW_2Q}")
    print(f"{'='*70}")
    run_e8a(simulator, groups, n_epochs=epochs)

    print(f"\n{'='*70}")
    print(f"E8b — Noise sweep (D+E) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    run_e8b(simulator, groups, n_epochs=epochs)

    print(f"\n{'='*70}")
    print(f"E8c — Cross-system (D+E) | n_per={n_per} | epochs={epochs}")
    print(f"{'='*70}")
    run_e8c(simulator, groups, n_epochs=epochs)

    print("\nE8 completed. Results in experiments/results/e8_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(fast=args.fast)
