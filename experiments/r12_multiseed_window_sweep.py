"""R12 — multi-seed replication of E2 (window-length sweep, scenarios B+C).

Why
---
E2 is a single run. Its AUROC series 0.933, 0.950, 0.947, 0.872, 0.969,
0.990 has one point (f = 0.20) out of line by 0.08-0.10, on a test set of
24 trajectories. Before explaining it, check whether it is there at all.

Writes ONLY to
experiments/results/r12_multiseed_window_sweep[_<tag>][_fast].csv.

Usage
-----
    python -m experiments.r12_multiseed_window_sweep --seeds 42 43 44
    python -m experiments.r12_multiseed_window_sweep --fast
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from experiments import common
from experiments import e2_window_sweep

COLUMNS = ("seed", "f", "r2", "mae", "auroc", "n_samples")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E2 for each seed and writes the rows together."""
    args = common.parse_args(__doc__, argv=argv)
    sizes = {"n_per": 30, "n_epochs": 10} if args.fast else {}
    path = common.results_path("r12_multiseed_window_sweep", args)
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds, config=sizes) as out:
        for seed in args.seeds:
            start = time.time()
            for result in e2_window_sweep.run(seed=seed, **sizes):
                out.write({
                    "seed": seed, "f": result["window_fraction"],
                    "r2": result["r2"], "mae": result["mae"],
                    "auroc": result["auroc"],
                    "n_samples": result["n_samples"]})
            print(f"R12 | seed {seed} done [{time.time() - start:.0f}s]",
                  flush=True)


if __name__ == "__main__":
    main()
