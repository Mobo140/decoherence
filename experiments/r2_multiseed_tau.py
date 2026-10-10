"""R2 — multi-seed variance probe for the adaptive-prior tau sweep (E7b shape).

Paper 2 concludes "the optimal threshold is tau=0.5 for XXZ" from a single
sweep in which adjacent tau values differ by up to 0.46 in R^2 -- larger than
the effect being claimed.  E7b also generates the dataset once per scenario and
trains every tau on it, while torch is never seeded, so consecutive tau points
differ in weight initialisation as well as in tau.  This re-runs the sweep
across seeds so the tau effect can be separated from run-to-run spread.

Writes ONLY to experiments/results/r2_multiseed_tau.csv; the champion
e7b_tau_sweep.csv is left untouched.

Reduced size (default 100 trajectories / 30 epochs) versus the champion
configuration, so absolute values are not comparable to the paper table -- the
spread, not the level, is the point.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.data.scenarios import build_configs, trajectory_store
from src.evaluation import stats
from src.models import registry
from src.simulation.simulator import QuTipSimulator

TAUS = [0.0, 0.3, 0.5, 0.7, 0.9]
OUT = Path(__file__).parent / "results" / "r2_multiseed_tau.csv"  # overridable via --out


def summarise(rows: list, scenarios: list) -> None:
    """Prints mean and standard deviation of R^2 and AUROC per tau.

    Args:
        rows: Result rows with the keys scenario, tau, r2 and auroc.
        scenarios: Scenarios to print, in order.
    """
    print(f"\n{'='*62}\nMEAN +/- STD PER TAU\n{'='*62}")
    print(f"{'scen':5} {'tau':>5} {'R2 mean':>9} {'R2 std':>8} "
          f"{'AUROC mean':>11} {'AUROC std':>10}")
    for scen in scenarios:
        for tau in TAUS:
            sub = [r for r in rows
                   if r["scenario"] == scen and r["tau"] == tau]
            r2_mean, r2_std = stats.mean_std([r["r2"] for r in sub])
            au_mean, au_std = stats.mean_std([r["auroc"] for r in sub])
            print(f"{scen:5} {tau:5.1f} {r2_mean:9.3f} {r2_std:8.3f} "
                  f"{au_mean:11.3f} {au_std:10.3f}")


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

    print(f"R2 tau sweep x seeds={a.seeds} n_per={a.n_per} epochs={a.epochs}", flush=True)
    sim, rows, t0 = QuTipSimulator(), [], time.time()

    for seed in a.seeds:
        groups = build_configs(n_per_scenario=a.n_per, seed=seed)
        for scen in a.scenarios:
            ds = GenerateDatasetUseCase(sim, store=trajectory_store()).execute(
                GenerateDatasetCommand(configs=groups[scen], window_length=20,
                                       horizon=1.0, samples_per_trajectory=5, seed=seed))
            for tau in TAUS:
                pred = registry.build_model(
                    "physics_lstm", hidden_size=128, num_layers=2,
                    dropout=0.3, adaptive_prior_r2_threshold=tau)
                TrainModelUseCase(pred).execute(TrainModelCommand(
                    dataset=ds, n_epochs=a.epochs, batch_size=64,
                    seed=seed,
                    learning_rate=5e-4, regression_loss="huber", verbose=False))
                m = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
                row = {"seed": seed, "scenario": scen, "tau": tau,
                       "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc}
                rows.append(row)
                OUT.parent.mkdir(exist_ok=True)
                with OUT.open("w" if len(rows) == 1 else "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if len(rows) == 1:
                        w.writeheader()
                    w.writerow(row)
                print(f"  seed={seed} {scen} tau={tau:.1f} R2={m.r2:8.3f} "
                      f"AUROC={m.risk_auroc:.3f}", flush=True)

    summarise(rows, a.scenarios)
    print(f"\ntotal {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
