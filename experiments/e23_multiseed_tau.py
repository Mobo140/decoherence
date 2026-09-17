"""E23 — multi-seed variance probe for the adaptive-prior tau sweep (E7b shape).

Paper 2 concludes "the optimal threshold is tau=0.5 for XXZ" from a single
sweep in which adjacent tau values differ by up to 0.46 in R^2 -- larger than
the effect being claimed.  E7b also generates the dataset once per scenario and
trains every tau on it, while torch is never seeded, so consecutive tau points
differ in weight initialisation as well as in tau.  This re-runs the sweep
across seeds so the tau effect can be separated from run-to-run spread.

Writes ONLY to experiments/results/e23_multiseed_tau.csv; the champion
e7b_tau_sweep.csv is left untouched.

Reduced size (default 100 trajectories / 30 epochs) versus the champion
configuration, so absolute values are not comparable to the paper table -- the
spread, not the level, is the point.
"""
from __future__ import annotations

import argparse, csv, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

TAUS = [0.0, 0.3, 0.5, 0.7, 0.9]
OUT = Path(__file__).parent / "results" / "e23_multiseed_tau.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-per", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--scenarios", nargs="+", default=["D", "E"])
    a = p.parse_args()

    print(f"E23 tau sweep x seeds={a.seeds} n_per={a.n_per} epochs={a.epochs}", flush=True)
    sim, rows, t0 = QuTipSimulator(), [], time.time()

    for seed in a.seeds:
        groups = build_configs(n_per_scenario=a.n_per, seed=seed)
        for scen in a.scenarios:
            ds = GenerateDatasetUseCase(sim, store=trajectory_store()).execute(
                GenerateDatasetCommand(configs=groups[scen], window_length=20,
                                       horizon=1.0, samples_per_trajectory=5, seed=seed))
            for tau in TAUS:
                pred = LSTMPredictor(hidden_size=128, num_layers=2, dropout=0.3,
                                     use_physics_prior=True,
                                     adaptive_prior_r2_threshold=tau)
                TrainModelUseCase(pred).execute(TrainModelCommand(
                    dataset=ds, n_epochs=a.epochs, batch_size=64,
                    learning_rate=5e-4, regression_loss="huber", verbose=False))
                m = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
                rows.append({"seed": seed, "scenario": scen, "tau": tau,
                             "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc})
                print(f"  seed={seed} {scen} tau={tau:.1f} R2={m.r2:8.3f} "
                      f"AUROC={m.risk_auroc:.3f}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{'='*62}\nMEAN +/- STD PER TAU\n{'='*62}")
    print(f"{'scen':5} {'tau':>5} {'R2 mean':>9} {'R2 std':>8} {'AUROC mean':>11} {'AUROC std':>10}")
    for scen in a.scenarios:
        for tau in TAUS:
            sub = [r for r in rows if r["scenario"] == scen and r["tau"] == tau]
            r2s = [r["r2"] for r in sub]; aus = [r["auroc"] for r in sub]
            sd = statistics.stdev(r2s) if len(r2s) > 1 else 0.0
            sa = statistics.stdev(aus) if len(aus) > 1 else 0.0
            print(f"{scen:5} {tau:5.1f} {statistics.mean(r2s):9.3f} {sd:8.3f} "
                  f"{statistics.mean(aus):11.3f} {sa:10.3f}")
    print(f"\ntotal {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
