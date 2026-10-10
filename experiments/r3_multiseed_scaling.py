"""R3 — multi-seed check of the two E11 claims Paper 2 marks as hypotheses.

Why
---
Paper 2 carries two scaling statements that were never replicated:

  * "physics_lstm degrades as data grows" — E11b (1000 trajs) scores
    cross-system XXZ R²=0.491 against E8c's 0.740 (500 trajs).
  * "E11a trades R² for AUROC" — E11a scores R²=0.566 / AUROC=0.888 on
    TFIM against E7a's 0.599 / 0.736.

Neither comparison isolates the factor it names.  E8c and E11b differ in
window length (30 vs 20), early-stopping patience and inference codes as
well as in trajectory count; E7a and E11a differ in window length (50 vs
20), min-gap and the whole evaluation protocol.  A difference between two
such runs cannot be attributed to data size or to window length, and a
single run cannot separate either from seed noise.

Design
------
Four arms, three seeds each, every arm evaluated through the *same*
protocol (E11's ``_eval_per_scenario``), so arms differ in one factor:

    e11a_w20    Transformer, window 20, 500 trajs   (published E11a)
    e7a_w50     Transformer, window 50, 500 trajs   <- window contrast
    e11b_500    physics_lstm E11b config, 500 trajs <- data-size contrast
    e11b_1000   physics_lstm E11b config, 1000 trajs (published E11b)

The arms call ``run_e11a`` / ``run_e11b`` directly rather than copying
their configuration, so the probe cannot drift away from the experiment
it replicates.  ``e7a_w50`` is *not* E7a: it applies E7a's window inside
E11a's protocol, and so isolates the window without reproducing E7a's
published number.

Writes ONLY to experiments/results/r3_multiseed_scaling.csv; the
champion CSVs are never touched.

Usage
-----
    python -m experiments.r3_multiseed_scaling --fast
    python -m experiments.r3_multiseed_scaling --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from experiments.e11_scaling import run_e11a, run_e11b
from src.data.scenarios import build_configs
from src.evaluation import stats
from src.simulation.simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r3_scratch.csv"   # run_e11* insist on writing a CSV


def _append(path: Path, row: dict, first: bool) -> None:
    """Append one row immediately so a long run survives interruption."""
    path.parent.mkdir(exist_ok=True)
    with path.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def summarise(rows: list) -> None:
    """Prints mean and standard deviation of R^2 and AUROC per arm.

    Args:
        rows: Result rows with the keys arm, scenario, r2 and auroc.
    """
    print("\n" + "=" * 62)
    print(f"{'arm':<11}{'scen':<5}{'R2 mean':>9}{'R2 std':>9}{'AUROC':>8}"
          f"{'std':>7}{'n':>4}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["scenario"])].append(r)
    for key in sorted(groups):
        r2_mean, r2_std = stats.mean_std(
            [float(x["r2"]) for x in groups[key]])
        au_mean, au_std = stats.mean_std(
            [float(x["auroc"]) for x in groups[key]])
        print(f"{key[0]:<11}{key[1]:<5}{r2_mean:9.3f}{r2_std:9.3f}"
              f"{au_mean:8.3f}{au_std:7.3f}{len(groups[key]):4d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r3_multiseed_scaling.csv")
    p.add_argument("--arms", nargs="+",
                   default=["e11a_w20", "e7a_w50", "e11b_500", "e11b_1000"])
    a = p.parse_args()

    n_500 = 50 if a.fast else 500
    n_1000 = 100 if a.fast else 1000
    epochs = 10 if a.fast else 150
    out = RESULTS_DIR / a.out

    print(f"R3 | seeds={a.seeds} arms={a.arms} n={n_500}/{n_1000} epochs={epochs}")
    print(f"writing -> {out}  (champion CSVs untouched)\n", flush=True)

    sim, first, t0 = QuTipSimulator(), True, time.time()
    collected = []

    for seed in a.seeds:
        g500 = build_configs(n_per_scenario=n_500, seed=seed)
        g1000 = build_configs(n_per_scenario=n_1000, seed=seed)
        for arm in a.arms:
            ts = time.time()
            print(f"--- seed {seed} | {arm} ---", flush=True)
            if arm == "e11a_w20":
                rows = run_e11a(sim, g500, n_epochs=epochs, n_per=n_500,
                                verbose=False, seed=seed, window=20, out_path=SCRATCH)
            elif arm == "e7a_w50":
                rows = run_e11a(sim, g500, n_epochs=epochs, n_per=n_500,
                                verbose=False, seed=seed, window=50, out_path=SCRATCH)
            elif arm == "e11b_500":
                rows = run_e11b(sim, g500, n_epochs=epochs,
                                verbose=False, seed=seed, out_path=SCRATCH)
            elif arm == "e11b_1000":
                rows = run_e11b(sim, g1000, n_epochs=epochs,
                                verbose=False, seed=seed, out_path=SCRATCH)
            else:
                raise ValueError(f"unknown arm: {arm}")

            for r in rows:
                row = {"seed": seed, "arm": arm, "scenario": r["scenario"],
                       "r2": r["r2"], "mae": r["mae"], "auroc": r["auroc"],
                       "n_samples": r["n_samples"]}
                _append(out, row, first)
                first = False
                collected.append(row)
                print(f"    {arm:<10} {r['scenario']} R2={r['r2']:8.3f} "
                      f"AUROC={r['auroc']:.3f}", flush=True)
            print(f"    [{time.time() - ts:.0f}s]", flush=True)

    SCRATCH.unlink(missing_ok=True)

    summarise(collected)
    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
