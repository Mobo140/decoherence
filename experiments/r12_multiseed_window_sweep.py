"""R12 — multi-seed replication of E2 (window-length sweep, scenarios B+C).

Why
---
E2 is a single run. Its AUROC series 0.933, 0.950, 0.947, 0.872, 0.969,
0.990 has one point (f = 0.20) out of line by 0.08-0.10, on a test set of
24 trajectories. Before explaining it, check whether it is there at all.

Writes ONLY to experiments/results/r12_multiseed_window_sweep.csv.

Usage
-----
    python -m experiments.r12_multiseed_window_sweep --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.e2_window_sweep as e2

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r12_scratch.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=str, default="r12_multiseed_window_sweep.csv")
    a = p.parse_args()
    out = RESULTS_DIR / a.out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "f", "r2", "mae", "auroc", "n_samples"])
        w.writeheader()
        for seed in a.seeds:
            ts = time.time()
            for r in e2.main(seed=seed, out_path=SCRATCH):
                w.writerow({"seed": seed, "f": r["window_fraction"], "r2": r["r2"],
                            "mae": r["mae"], "auroc": r["auroc"],
                            "n_samples": r["n_samples"]})
                f.flush()
            print(f"R12 | seed {seed} done [{time.time() - ts:.0f}s]", flush=True)
    SCRATCH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
