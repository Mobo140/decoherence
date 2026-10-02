"""R10 — does the TFIM J-dependence disappear with an energy-basis target?

Why
---
R9 found that the T2 target, the first 1/e crossing of the l1-coherence in
the computational basis, is modulated by the TFIM Hamiltonian with a depth
set by tan(2 theta) = J / 2h, and that the share of trajectories whose
coherence re-crosses the threshold tracks the R5 loss in R^2 across J
(r = -0.85). That is a correlation over seven points.

This is the direct test: repeat R5 exactly (same configurations, seeds,
model, training) and change only the target, to the 1/e crossing of the
l1-coherence in the eigenbasis of H, which unitary evolution does not
modulate. The model inputs stay in the computational basis.

  * If R^2(J) flattens, the TFIM ceiling is a property of the target.
  * If the decline persists, its cause is in the dynamics.

The comparison is paired with R5 by seed (same config_seed, same seed).

Writes ONLY to experiments/results/r10_energy_target_J_sweep.csv.

Usage
-----
    python -m experiments.r10_energy_target_J_sweep --fast
    python -m experiments.r10_energy_target_J_sweep --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.e7_paper2_extended import run_e7c
from src.domain.value_objects import DecoherenceCriterion
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r10_scratch.csv"
R5 = RESULTS_DIR / "r5_multiseed_J_sweep.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r10_energy_target_J_sweep.csv")
    a = p.parse_args()

    n_per_J = 30 if a.fast else 200
    epochs = 10 if a.fast else 150
    out = RESULTS_DIR / a.out
    print(f"R10 | seeds={a.seeds} n_per_J={n_per_J} epochs={epochs}", flush=True)

    sim, collected = QuTipSimulator(), []
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "J", "r2", "mae", "auroc", "n_samples"])
        w.writeheader()
        for seed in a.seeds:
            ts = time.time()
            rows = run_e7c(sim, n_per_J=n_per_J, n_epochs=epochs, verbose=False,
                           seed=seed, config_seed=seed, out_path=SCRATCH,
                           criterion=DecoherenceCriterion.COHERENCE_ENERGY)
            for r in rows:
                row = {"seed": seed, "J": r["J"], "r2": r["r2"], "mae": r["mae"],
                       "auroc": r["auroc"], "n_samples": r["n_samples"]}
                w.writerow(row)
                f.flush()
                collected.append(row)
                print(f"R10 | seed={seed} J={r['J']:.1f} R2={r['r2']:.3f} "
                      f"AUROC={r['auroc']:.3f}", flush=True)
            print(f"R10 | seed {seed} done [{time.time() - ts:.0f}s]", flush=True)
    SCRATCH.unlink(missing_ok=True)

    if a.fast or not R5.exists():
        return
    base = defaultdict(dict)
    with R5.open() as f:
        for r in csv.DictReader(f):
            base[float(r["J"])][int(r["seed"])] = float(r["r2"])
    new = defaultdict(dict)
    for r in collected:
        new[float(r["J"])][int(r["seed"])] = float(r["r2"])

    def drop(d):
        lo = [v for J, s in d.items() if J <= 0.5 for v in s.values()]
        hi = [v for J, s in d.items() if J >= 0.8 for v in s.values()]
        return st.mean(lo) - st.mean(hi)

    print(f"\n{'J':>5}{'R5 R2':>10}{'R10 R2':>10}{'paired diff':>14}")
    for J in sorted(new):
        seeds = sorted(set(new[J]) & set(base[J]))
        d = [new[J][s] - base[J][s] for s in seeds]
        print(f"{J:>5.1f}{st.mean(base[J].values()):>10.3f}"
              f"{st.mean(new[J].values()):>10.3f}{st.mean(d):>+14.3f}")
    print(f"\ndrop (J<=0.5 minus J>=0.8): R5 {drop(base):+.3f}  R10 {drop(new):+.3f}")


if __name__ == "__main__":
    main()
