"""R1 — multi-seed variance probe for the two-qubit ablation (E6a shape).

Why this exists
---------------
Every number in both papers comes from a single run with seed=42, so there are
no error bars anywhere.  Several conclusions rest on small gaps (e.g.
physics_lstm 0.636 vs lstm_only 0.673 on XXZ, a difference of 0.037), and the
E7b tau sweep swings by up to 0.46 between adjacent tau values, which is larger
than the effect it claims to measure.  This script re-runs the E6a-shaped
ablation across several seeds and reports mean +/- std so those gaps can be
judged against run-to-run spread.

Safety
------
Writes ONLY to experiments/results/r1_multiseed_ablation.csv.  It never touches
the champion CSVs (e6_*_improved.csv etc.) that back the paper tables --
re-running e6_2qubit_improved.py directly would overwrite them, which is why
this is a separate entry point.

Caveat on interpretation
------------------------
Runs at a reduced size (default n_per=100, epochs=30) rather than the champion
configuration (500 / 150), so absolute values are NOT comparable to the paper
tables.  With less data and less training the spread here is, if anything, an
*upper bound* on the spread of the full configuration.  Use it to judge whether
a gap is inside the noise, not to restate headline numbers.

Each seed varies both the sampled physical configurations (build_configs) and
the training/split randomness (AblationCommand), i.e. it is a full experimental
replication rather than a training-only reseed.

Usage
-----
    python -m experiments.r1_multiseed_ablation --seeds 42 43 44
    python -m experiments.r1_multiseed_ablation --seeds 42 43 44 --n-per 150 --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from src.application.run_ablation import AblationCommand, AblationStudyUseCase
from src.domain.value_objects import PredictorVariant
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

class ImprovedAblation(AblationStudyUseCase):
    """Mirror of the predictor factory used by E6a (``e6_2qubit_improved.py``).

    The base use case builds default predictors (hidden 64, 1 layer, tau=0),
    which is a *different, weaker* model than the one the paper reports.
    Replicating E6a's spread requires E6a's architecture, so the factory is
    copied here verbatim; keep the two in sync if either changes.
    """

    def _build_predictor(self, variant, command):
        from src.infrastructure.ml.lstm_predictor import LSTMPredictor
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        from src.infrastructure.ml.transformer_predictor import TransformerPredictor
        if variant == PredictorVariant.PHYSICS_ONLY:
            return PhysicsOnlyPredictor(t_max=max(c.t_max for c in command.configs))
        if variant == PredictorVariant.LSTM_ONLY:
            return LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=False, adaptive_prior_r2_threshold=0.0,
            )
        if variant == PredictorVariant.PHYSICS_LSTM:
            return LSTMPredictor(
                hidden_size=128, num_layers=2, dropout=0.3,
                use_physics_prior=True, adaptive_prior_r2_threshold=0.7,
            )
        if variant == PredictorVariant.TRANSFORMER:
            return TransformerPredictor()
        raise ValueError(variant)


VARIANTS = [
    PredictorVariant.PHYSICS_ONLY,
    PredictorVariant.LSTM_ONLY,
    PredictorVariant.PHYSICS_LSTM,
    PredictorVariant.TRANSFORMER,
]
OUT = Path(__file__).parent / "results" / "r1_multiseed_ablation.csv"  # overridable via --out


def _append(row: dict, first: bool) -> None:
    """Append one row immediately so a long run survives interruption."""
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def run_seed(simulator, seed: int, n_per: int, epochs: int, scenarios: list,
             first_row: list = None) -> list:
    groups = build_configs(n_per_scenario=n_per, seed=seed)
    uc = ImprovedAblation(simulator=simulator)
    rows = []
    for scen in scenarios:
        cmd = AblationCommand(
            configs=groups[scen],
            variants=VARIANTS,
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=epochs,
            batch_size=64,
            lr=5e-4,
            regression_loss="huber",
            seed=seed,
            verbose=False,
        )
        for r in uc.execute(cmd):
            row = {
                "seed": seed,
                "scenario": scen,
                "variant": r.spec.predictor_variant.value,
                "r2": r.r2,
                "mae": r.mae,
                "auroc": r.risk_auroc,
                "n_samples": r.n_samples,
            }
            rows.append(row)
            _append(row, first_row[0])
            first_row[0] = False
            print(f"    seed={seed} {scen} {r.spec.predictor_variant.value:14} "
                  f"R2={r.r2:8.3f} MAE={r.mae:7.3f} AUROC={r.risk_auroc:.3f}", flush=True)
    return rows


def summarise(rows: list) -> None:
    print(f"\n{'='*78}\nMEAN +/- STD ACROSS SEEDS\n{'='*78}")
    print(f"{'scen':5} {'variant':14} {'R2 mean':>9} {'R2 std':>8} "
          f"{'AUROC mean':>11} {'AUROC std':>10} {'n':>3}")
    for scen in sorted({r["scenario"] for r in rows}):
        for var in [v.value for v in VARIANTS]:
            sub = [r for r in rows if r["scenario"] == scen and r["variant"] == var]
            if not sub:
                continue
            r2s = [r["r2"] for r in sub]
            aus = [r["auroc"] for r in sub]
            sd = statistics.stdev(r2s) if len(r2s) > 1 else 0.0
            sda = statistics.stdev(aus) if len(aus) > 1 else 0.0
            print(f"{scen:5} {var:14} {statistics.mean(r2s):9.3f} {sd:8.3f} "
                  f"{statistics.mean(aus):11.3f} {sda:10.3f} {len(sub):3d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-per", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--scenarios", nargs="+", default=["D", "E"])
    p.add_argument("--out", type=str, default=None,
                   help="output CSV name under experiments/results/")
    a = p.parse_args()
    global OUT
    if a.out:
        OUT = Path(__file__).parent / "results" / a.out

    print(f"R1 multi-seed variance | seeds={a.seeds} n_per={a.n_per} "
          f"epochs={a.epochs} scenarios={a.scenarios}")
    print(f"writing -> {OUT}  (champion CSVs untouched)\n")

    sim = QuTipSimulator()
    all_rows, t0, first = [], time.time(), [True]
    for i, s in enumerate(a.seeds, 1):
        print(f"--- seed {s}  ({i}/{len(a.seeds)}) ---", flush=True)
        ts = time.time()
        all_rows += run_seed(sim, s, a.n_per, a.epochs, a.scenarios, first)
        print(f"    [{time.time()-ts:.0f}s]", flush=True)

    summarise(all_rows)
    print(f"\ntotal {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
