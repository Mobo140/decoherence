"""R5 — is the predictability dip near the TFIM transition reproducible?

Why
---
Paper 2 reports a pronounced dip in prediction accuracy around the TFIM
quantum phase transition (worst at J=1.2, R^2=0.319, against 0.876 at
J=0.2 and 0.836 at J=1.5) and argues it is structural: the
amplitude-damping dissipator does not commute with the parity operator,
so it mixes the two symmetry sectors, most damagingly where their
timescales are comparable.

The argument is worked out in detail, but its entire empirical basis is
one sweep of seven J values at a single seed, with 81-94 evaluation
samples per point. Everywhere else in this pipeline that a single run
looked decisive, replication has either reversed it or dissolved it, so
the dip needs the same treatment before the structural reading can be
relied on.

What this measures
------------------
The same sweep across seeds, varying both the sampled configurations
(config_seed) and training (seed). This answers whether the dip is
there at all and whether its location is stable; it does not test the
proposed mechanism, which would need a dissipator that commutes with
parity.

Writes ONLY to experiments/results/r5_multiseed_J_sweep.csv.

Usage
-----
    python -m experiments.r5_multiseed_J_sweep --fast
    python -m experiments.r5_multiseed_J_sweep --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.e7_paper2_extended import run_e7c
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r5_scratch.csv"
H_FIELD = 1.0          # TwoQubitSystem hardcodes h = 1.0


def _append(path: Path, row: dict, first: bool) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def _stats(values):
    """Mean and s.d. over the finite entries; AUROC is NaN when a bucket has
    only one risk class, and a degenerate point must not kill the summary."""
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan"), 0
    sd = st.stdev(good) if len(good) > 1 else 0.0
    return st.mean(good), sd, len(good)


def _r(J: float) -> float:
    """Parity-splitting ratio the structural argument is built on."""
    return J / math.sqrt(J * J + 4 * H_FIELD * H_FIELD)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r5_multiseed_J_sweep.csv")
    a = p.parse_args()

    n_per_J = 30 if a.fast else 200
    epochs = 10 if a.fast else 150
    out = RESULTS_DIR / a.out

    print(f"R5 | seeds={a.seeds} n_per_J={n_per_J} epochs={epochs}")
    print(f"writing -> {out}  (champion CSVs untouched)\n", flush=True)

    sim, first, t0, collected = QuTipSimulator(), True, time.time(), []

    for seed in a.seeds:
        ts = time.time()
        print(f"--- seed {seed} ---", flush=True)
        rows = run_e7c(sim, n_per_J=n_per_J, n_epochs=epochs, verbose=False,
                       seed=seed, config_seed=seed, out_path=SCRATCH)
        for r in rows:
            row = {"seed": seed, "J": r["J"], "r2": r["r2"], "mae": r["mae"],
                   "auroc": r["auroc"], "n_samples": r["n_samples"]}
            _append(out, row, first)
            first = False
            collected.append(row)
            print(f"    J={r['J']:.1f} R2={r['r2']:8.3f} AUROC={r['auroc']:.3f}",
                  flush=True)
        print(f"    [{time.time() - ts:.0f}s]", flush=True)

    SCRATCH.unlink(missing_ok=True)

    g = defaultdict(list)
    for r in collected:
        g[float(r["J"])].append(r)

    print("\n" + "=" * 66)
    print(f"{'J':>5}{'r(J)':>7}{'R2 mean':>10}{'±std':>8}{'AUROC':>8}{'±std':>7}"
          f"{'published':>11}")
    published = {0.2: 0.876, 0.5: 0.770, 0.8: 0.469, 1.0: 0.497,
                 1.2: 0.319, 1.5: 0.836, 2.0: 0.614}
    for J in sorted(g):
        m_r2, sd, n_r2 = _stats([float(x["r2"]) for x in g[J]])
        m_au, sa, n_au = _stats([float(x["auroc"]) for x in g[J]])
        flag = "" if n_au == len(g[J]) else f"  (AUROC n={n_au})"
        print(f"{J:>5.1f}{_r(J):>7.3f}{m_r2:>10.3f}{sd:>8.3f}"
              f"{m_au:>8.3f}{sa:>7.3f}{published.get(J, float('nan')):>11.3f}{flag}")

    means = {J: _stats([float(x["r2"]) for x in v])[0] for J, v in g.items()}
    means = {J: m for J, m in means.items() if math.isfinite(m)}
    if means:
        worst = min(means, key=means.get)
        best = max(means, key=means.get)
        spread = means[best] - means[worst]
        sds = [_stats([float(x["r2"]) for x in v])[1]
               for v in g.values() if len(v) > 1]
        sds = [x for x in sds if math.isfinite(x)]
        typical = st.mean(sds) if sds else 0.0
        print(f"\nworst mean at J={worst} ({means[worst]:.3f}), "
              f"best at J={best} ({means[best]:.3f})")
        print(f"dip depth {spread:.3f} vs typical seed spread {typical:.3f}"
              f"  -> {'resolved' if spread > 2 * typical else 'NOT resolved'}")
        per_seed_worst = {}
        for r in collected:
            if not math.isfinite(float(r["r2"])):
                continue
            s = r["seed"]
            if s not in per_seed_worst or float(r["r2"]) < float(per_seed_worst[s]["r2"]):
                per_seed_worst[s] = r
        print("worst J per seed: "
              + ", ".join(f"seed {s}: J={float(v['J']):.1f}"
                          for s, v in sorted(per_seed_worst.items())))

    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
