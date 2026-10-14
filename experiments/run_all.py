"""Master entry point: runs E1-E5 sequentially with their defaults.

Usage
-----
    python -m experiments.run_all [--fast]

--fast runs every experiment on reduced data and writes `*_fast.csv`:
enough to verify the pipeline, not publication-quality results.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from experiments import common
from experiments import e1_ablation
from experiments import e2_window_sweep
from experiments import e3_noise_sweep
from experiments import e4_cross_system
from experiments import e5_inverse_baseline


def _log(message: str) -> None:
    print(f"\n{'=' * 70}\n{message}\n{'=' * 70}")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E1-E5."""
    args = common.parse_fast(__doc__, argv)
    fast = ["--fast"] if args.fast else []
    start = time.time()
    for title, experiment in (
        ("E1: Ablation study", e1_ablation),
        ("E2: Window fraction sweep", e2_window_sweep),
        ("E3: Noise robustness sweep", e3_noise_sweep),
        ("E4: Cross-system generalisation", e4_cross_system),
        ("E5: Inverse-problem baseline comparison", e5_inverse_baseline),
    ):
        _log(title)
        experiment.main(fast)
    print(f"\nAll experiments completed in "
          f"{(time.time() - start) / 60:.1f} min.")
    print("Results written to experiments/results/")


if __name__ == "__main__":
    main()
