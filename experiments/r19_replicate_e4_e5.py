"""R19 — E4 (one model for A+B+C) and E5 (direct vs inverse) across seeds.

Both were single runs. This calls e4.main / e5.main with seeds 42-44 and
writes experiments/results/r19_e4_s<seed>.csv and r19_e5_s<seed>.csv.

Usage
-----
    python -m experiments.r19_replicate_e4_e5 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.e4_cross_system as e4
import experiments.e5_inverse_baseline as e5

RES = Path(__file__).parent / "results"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = p.parse_args()
    for s in a.seeds:
        e4.main(train_scenarios=["A", "B", "C"], eval_scenarios=["A", "B", "C"],
                seed=s, out_path=RES / f"r19_e4_s{s}.csv")
        e5.main(seed=s, out_path=RES / f"r19_e5_s{s}.csv")
        print(f"R19 | seed {s} done", flush=True)
