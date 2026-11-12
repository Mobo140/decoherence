"""R19 — E4 (one model for A+B+C) and E5 (direct vs inverse) across seeds.

Both were single runs. This runs E4 and E5 with seeds 42-44 and writes
experiments/results/r19_e4_s<seed>.csv and r19_e5_s<seed>.csv, one pair
per seed.

Usage
-----
    python -m experiments.r19_replicate_e4_e5 --seeds 42 43 44
    python -m experiments.r19_replicate_e4_e5 --fast --seeds 42
"""
from __future__ import annotations

from collections.abc import Sequence

from experiments import common
from experiments import e4_cross_system
from experiments import e5_inverse_baseline


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E4 and E5 for each seed."""
    args = common.parse_args(__doc__, argv=argv, out=False)
    sizes_e4 = {"n_per": 30, "n_epochs": 10} if args.fast else {}
    sizes_e5 = (
        {"n_per": 20, "n_epochs": 10, "recon_epochs": 10} if args.fast
        else {})
    suffix = "_fast" if args.fast else ""
    tag = f"_{args.tag}" if args.tag else ""
    for seed in args.seeds:
        scenarios = ("A", "B", "C")
        rows = e4_cross_system.run(
            scenarios, scenarios, seed=seed, **sizes_e4)
        path = common.RESULTS_DIR / f"r19_e4{tag}_s{seed}{suffix}.csv"
        common.write_results(
            rows, path, columns=e4_cross_system.COLUMNS, seeds=[seed],
            config={"experiment": "E4", **sizes_e4})
        rows = e5_inverse_baseline.run(seed=seed, **sizes_e5)
        path = common.RESULTS_DIR / f"r19_e5{tag}_s{seed}{suffix}.csv"
        common.write_results(
            rows, path, columns=e5_inverse_baseline.COLUMNS, seeds=[seed],
            config={"experiment": "E5", **sizes_e5})
        print(f"R19 | seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
