"""R7 — multi-seed replication of Paper 1's ablation (E1, scenarios A-C).

Why
---
Paper 1 was never put through the check that dismantled five of Paper 2's
conclusions. Its numbers are single runs too.

Sorting its claims by margin shows where the risk actually sits. The
central one is not at risk: lstm_only scores -0.446, 0.009 and -0.029
against roughly 0.95 for physics_lstm, so "the physics prior is
essential" has a margin of order 1.0 in R^2. Neither is C1
(physics_only = 0.9995 on the constant-gamma scenario, an analytic
result), nor physics_lstm over physics_only on B (+0.19), nor the
residual over the Transformer on A (+0.18).

Two comparisons do rest on thin margins, both on scenario C:

    physics_lstm 0.9756 vs physics_only 0.9649   -> +0.011
    transformer  0.9808 vs physics_lstm  0.9756  -> +0.005

For scale, replication dissolved margins of 0.037 and 0.04 in Paper 2.

This probe calls e1_ablation.main rather than rebuilding the command, so
it cannot drift from the experiment it replicates -- E1 uses the base
AblationStudyUseCase, whose default predictors are what Paper 1 reports,
so unlike R1 there is no override to mirror.

Writes ONLY to experiments/results/r7_multiseed_1qubit.csv.

Usage
-----
    python -m experiments.r7_multiseed_1qubit --fast
    python -m experiments.r7_multiseed_1qubit --seeds 42 43 44
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

import experiments.e1_ablation as e1

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r7_scratch.csv"

# Published single-run values (e1_ablation_p1.csv), for the comparison column.
PUBLISHED = {
    ("A", "physics_only"): 0.9995, ("A", "physics_lstm"): 0.9756,
    ("A", "lstm_only"): -0.4462,   ("A", "transformer"): 0.7991,
    ("B", "physics_only"): 0.7534, ("B", "physics_lstm"): 0.9449,
    ("B", "lstm_only"): 0.0094,    ("B", "transformer"): 0.9024,
    ("C", "physics_only"): 0.9649, ("C", "physics_lstm"): 0.9756,
    ("C", "lstm_only"): -0.0292,   ("C", "transformer"): 0.9808,
}

# The claims worth testing, as (scenario, better, worse, published margin).
CLAIMS = [
    ("A", "physics_only", "physics_lstm", 0.0239, "C1: formula exact at const gamma"),
    ("B", "physics_lstm", "physics_only", 0.1915, "C2: prior + residual beats formula (B)"),
    ("C", "physics_lstm", "physics_only", 0.0107, "C2: prior + residual beats formula (C)"),
    ("A", "physics_lstm", "lstm_only",    1.4218, "C2: prior is essential (A)"),
    ("B", "physics_lstm", "lstm_only",    0.9355, "C2: prior is essential (B)"),
    ("C", "physics_lstm", "lstm_only",    1.0048, "C2: prior is essential (C)"),
    ("A", "physics_lstm", "transformer",  0.1765, "residual preferred at const gamma"),
    ("C", "transformer",  "physics_lstm", 0.0052, "Transformer edge on C"),
]


def _append(path: Path, row: dict, first: bool) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r7_multiseed_1qubit.csv")
    a = p.parse_args()

    n_per = 30 if a.fast else 100
    out = RESULTS_DIR / a.out

    print(f"R7 | seeds={a.seeds} n_per={n_per} scenarios=A,B,C")
    print(f"writing -> {out}  (champion CSVs untouched)\n", flush=True)

    first, t0, collected = True, time.time(), []
    for seed in a.seeds:
        ts = time.time()
        print(f"--- seed {seed} ---", flush=True)
        rows = e1.main(scenarios=["A", "B", "C"], n_per=n_per,
                       seed=seed, out_path=SCRATCH)
        for r in rows:
            row = {"seed": seed, "scenario": r["scenario"],
                   "variant": r["variant"], "r2": r["r2"],
                   "mae": r["mae"], "auroc": r["auroc"]}
            _append(out, row, first)
            first = False
            collected.append(row)
        print(f"    [{time.time() - ts:.0f}s]", flush=True)

    SCRATCH.unlink(missing_ok=True)

    by = defaultdict(dict)
    for r in collected:
        by[(r["scenario"], r["variant"])][r["seed"]] = float(r["r2"])

    print("\n" + "=" * 66)
    print(f"{'scen':<5}{'variant':<15}{'R2 mean':>10}{'±std':>8}{'published':>11}")
    for key in sorted(by):
        v = list(by[key].values())
        sd = st.stdev(v) if len(v) > 1 else 0.0
        pub = PUBLISHED.get(key, float("nan"))
        print(f"{key[0]:<5}{key[1]:<15}{st.mean(v):>10.3f}{sd:>8.3f}{pub:>11.4f}")

    print("\nclaims, as paired differences per seed:")
    for scen, better, worse, pub_margin, label in CLAIMS:
        hi, lo = by.get((scen, better), {}), by.get((scen, worse), {})
        seeds = sorted(set(hi) & set(lo))
        if not seeds:
            continue
        d = [hi[s] - lo[s] for s in seeds]
        sd = st.stdev(d) if len(d) > 1 else 0.0
        if all(x > 0 for x in d):
            verdict = "holds in every seed"
        elif all(x < 0 for x in d):
            verdict = "REVERSED in every seed"
        else:
            verdict = "SIGN VARIES -- not resolved"
        print(f"  {label}")
        print(f"    published {pub_margin:+.4f} | replicated {st.mean(d):+.3f} "
              f"± {sd:.3f}  {[f'{x:+.3f}' for x in d]}  -> {verdict}")

    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
