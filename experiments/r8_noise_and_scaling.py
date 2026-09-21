"""R8 — replicate the two Paper 2 claims that never went through a seed check.

Why
---
Five of Paper 2's conclusions were overturned by replication; two were
never replicated at all, and both rest on single runs:

  Claim 2  scaling and improved training push TFIM cross-system R^2 from
           -0.381 (pilot) to +0.575 (E8c).
  Claim 6  noise robustness is exceptional -- R^2 spans a total range of
           only 0.007 across sigma = 0 to 0.10 (E6b).

Claim 6 is the one to worry about. It is a statement that a spread is
*small*, made from one run, in a pipeline whose seed-to-seed spread
reaches +/-0.36. If the 0.007 range is an accident of a single
initialisation it says nothing about robustness, exactly as the
"R^2 traded for AUROC" reading turned out to say nothing.

Claim 2's direction is not in doubt -- the gap is +0.96, the same order
as Paper 1's central claim -- but the champion value 0.575 is quoted
throughout both papers, so it is worth knowing its spread.

Arms (three seeds each):

    e6b   noise sweep, sigma in {0, 0.01, 0.02, 0.05, 0.10}
    e8c   cross-system physics_lstm, trained on D+E, evaluated per scenario

Both call the published functions rather than copying their configuration.

Writes ONLY to experiments/results/r8_noise_and_scaling.csv.

Usage
-----
    python -m experiments.r8_noise_and_scaling --fast
    python -m experiments.r8_noise_and_scaling --seeds 42 43 44
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

from experiments._config import build_configs
from experiments.e6_2qubit_improved import _noise_sweep
from experiments.e8_improved_2qubit import run_e8c
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r8_scratch.csv"

# Published R^2 from e6_noise_sweep_improved.csv (an earlier version of this
# file mistakenly held the AUROC column here).
PUBLISHED_E6B = {0.0: 0.5372, 0.01: 0.5383, 0.02: 0.5393, 0.05: 0.5440, 0.10: 0.5440}
PUBLISHED_E8C = {"D": 0.740, "E": 0.575}


def _append(path: Path, row: dict, first: bool) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def _stats(values):
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan")
    return st.mean(good), (st.stdev(good) if len(good) > 1 else 0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r8_noise_and_scaling.csv")
    p.add_argument("--arms", nargs="+", default=["e6b", "e8c"])
    a = p.parse_args()

    n_per = 50 if a.fast else 500
    epochs = 10 if a.fast else 150
    out = RESULTS_DIR / a.out

    print(f"R8 | seeds={a.seeds} arms={a.arms} n_per={n_per} epochs={epochs}")
    print(f"writing -> {out}  (champion CSVs untouched)\n", flush=True)

    sim, first, t0, collected = QuTipSimulator(), True, time.time(), []

    for seed in a.seeds:
        groups = build_configs(n_per_scenario=n_per, seed=seed)
        for arm in a.arms:
            ts = time.time()
            print(f"--- seed {seed} | {arm} ---", flush=True)
            if arm == "e6b":
                rows = _noise_sweep(sim, groups, ["D", "E"], epochs, "",
                                    verbose=False, seed=seed, out_path=SCRATCH)
                for r in rows:
                    row = {"seed": seed, "arm": arm, "key": f"{float(r['noise_sigma']):.2f}",
                           "r2": r["r2"], "mae": r["mae"], "auroc": r["auroc"]}
                    _append(out, row, first); first = False; collected.append(row)
            else:
                rows = run_e8c(sim, groups, n_epochs=epochs, verbose=False,
                               seed=seed, out_path=SCRATCH)
                for r in rows:
                    scen = "D" if "XXZ" in str(r.get("name", "")) or str(r.get("name", "")).endswith("D") else "E"
                    row = {"seed": seed, "arm": arm, "key": str(r.get("name", scen)),
                           "r2": r["r2"], "mae": r["mae"], "auroc": r["auroc"]}
                    _append(out, row, first); first = False; collected.append(row)
            print(f"    [{time.time() - ts:.0f}s]", flush=True)

    SCRATCH.unlink(missing_ok=True)

    g = defaultdict(list)
    for r in collected:
        g[(r["arm"], r["key"])].append(float(r["r2"]))

    print("\n" + "=" * 62)
    for arm in a.arms:
        keys = sorted(k for (am, k) in g if am == arm)
        if not keys:
            continue
        print(f"\n--- {arm} ---")
        print(f"{'key':<22}{'R2 mean':>10}{'±std':>9}{'published':>11}")
        for k in keys:
            m, sd = _stats(g[(arm, k)])
            pub = PUBLISHED_E6B.get(float(k), float("nan")) if arm == "e6b" \
                else next((v for kk, v in PUBLISHED_E8C.items() if kk in k), float("nan"))
            print(f"{k:<22}{m:>10.3f}{sd:>9.3f}{pub:>11.3f}")

        if arm == "e6b":
            # The claim is about the SPAN across sigma, measured within a run.
            per_seed = defaultdict(dict)
            for r in collected:
                if r["arm"] == "e6b":
                    per_seed[r["seed"]][r["key"]] = float(r["r2"])
            spans = [max(d.values()) - min(d.values()) for d in per_seed.values()]
            print(f"\n  span of R2 across sigma, per seed: "
                  f"{[f'{s:.3f}' for s in spans]}")
            print(f"  mean span {st.mean(spans):.3f} "
                  f"(published 0.007)")
            seed_sd = st.mean([_stats(g[('e6b', k)])[1] for k in keys])
            print(f"  typical seed-to-seed s.d. at fixed sigma: {seed_sd:.3f}")
            verdict = ("robustness claim holds: the sweep moves far less than the seeds do"
                       if st.mean(spans) < seed_sd else
                       "NOT supported: the sweep moves as much as the seeds do")
            print(f"  -> {verdict}")

    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
