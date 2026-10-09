"""R25 — learning curve for the entanglement-sudden-death target.

Why
---
R24 found T_ESD (first zero of the Wootters concurrence) predictable at
R^2 = 0.43-0.66 with 500 trajectories, while a 2% change of the rate moves
it by only 0.1-0.3%: the first death is set mostly by the state and the
Hamiltonian, which a window with the full state contains. That suggests
the remaining error is a learning problem rather than an information
limit. On XXZ the concurrence as an explicit input helped by 0.08 over the
correlators it is computed from, i.e. the network did not learn the
nonlinear map from 500 trajectories. Two predictions, if that reading is
right: R^2 keeps rising with the training set, and the gap between
"+ correlators" and "+ correlators + concurrence" closes.

Setup: Transformer (as in E11a/R24), window 20, scenarios D and E,
training sets of 250, 500, 1000, 2000 and 4000 trajectories, three input
sets (base / corr / corr_conc, as in R24), three seeds. The training sets
are nested: the 4000 trajectories are simulated once and each smaller set
is the first N of them (an in-memory store hands the same trajectories
back), so the curve is not confounded by resampling. Every model is
scored on the same 200 independent trajectories per scenario and seed as
in R24 (generator seed + 1000). Censored trajectories (still entangled at
t_max) are dropped, as everywhere.

Writes ONLY to experiments/results/r25_esd_data_scaling*.csv.

Usage
-----
    python -m experiments.r25_esd_data_scaling --fast
    python -m experiments.r25_esd_data_scaling --seeds 42 --scenario D --tag D42
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.r23_two_body_correlators import _all_test, _train, _transformer
from experiments.r24_entanglement_target import EntanglementSimulator, _esd, feature_set
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.domain.ports import ITrajectoryStore

RESULTS_DIR = Path(__file__).parent / "results"
SIZES = [250, 500, 1000, 2000, 4000]
FEATURES = ["base", "corr", "corr_conc"]


class MemoryStore(ITrajectoryStore):
    """Per-process trajectory store: the larger set is simulated first, so
    every smaller one is a sequence of hits and the initial-state stream is
    never shifted (cf. the partial-cache note in R23)."""

    def __init__(self):
        self._d = {}

    def save(self, trajectories, name):
        self._d[name] = list(trajectories)

    def load(self, name):
        if name not in self._d:
            raise FileNotFoundError(name)
        return self._d[name]

    def list_available(self):
        return list(self._d)


def run(seed, sc, sizes, n_eval, epochs, w, f):
    store = MemoryStore()
    gen = GenerateDatasetUseCase(EntanglementSimulator(), store=store)
    cfgs = _esd(build_configs(n_per_scenario=max(sizes), seed=seed))[sc]
    eval_cfgs = _esd(build_configs(n_per_scenario=n_eval, seed=seed + 1000))[sc]
    ts = time.time()
    ev = _all_test(gen.execute(GenerateDatasetCommand(
        configs=eval_cfgs, window_length=20, horizon=1.0, samples_per_trajectory=5,
        seed=seed + 1, train_ratio=0.0, val_ratio=0.0, dataset_name=f"r25_eval_{seed}_{sc}")))
    for n in sorted(sizes, reverse=True):          # largest first: smaller sets are cache hits
        tr = gen.execute(GenerateDatasetCommand(
            configs=cfgs[:n], window_length=20, horizon=1.0, samples_per_trajectory=5,
            seed=seed, dataset_name=f"r25_{seed}_{sc}_{n}"))
        print(f"R25 | seed={seed} {sc} n={n} windows={len(tr.sequences)} "
              f"censored={tr.metadata.get('n_censored')} [{time.time() - ts:.0f}s]", flush=True)
        for feats in FEATURES:
            pred = _train(_transformer(), feature_set(tr, feats), epochs, seed)
            m = BacktestUseCase(pred).execute(BacktestCommand(dataset=feature_set(ev, feats), horizon=1.0))
            w.writerow({"seed": seed, "scenario": sc, "n_traj": n, "features": feats,
                        "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc,
                        "n_samples": m.n_samples, "n_train_windows": len(tr.train_idx),
                        "n_censored": tr.metadata.get("n_censored")})
            f.flush()
            print(f"R25 | seed={seed} {sc} n={n} {feats:<9} R2={m.r2:.3f} MAE={m.mae:.3f} "
                  f"[{time.time() - ts:.0f}s]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--scenario", choices=["D", "E", "DE"], default="DE")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    sizes, n_eval, epochs = ([20, 40], 20, 5) if a.fast else (SIZES, 200, 150)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out = RESULTS_DIR / f"r25_esd_data_scaling{sfx}.csv"
    print(f"R25 | seeds={a.seeds} scenarios={a.scenario} sizes={sizes}", flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "scenario", "n_traj", "features", "r2", "mae",
                                          "auroc", "n_samples", "n_train_windows", "n_censored"])
        w.writeheader()
        for seed in a.seeds:
            for sc in a.scenario:
                run(seed, sc, sizes, n_eval, epochs, w, f)


if __name__ == "__main__":
    main()
