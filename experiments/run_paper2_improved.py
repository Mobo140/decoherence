"""Paper 2 improved — Two-qubit early warning with roadmap fixes applied.

Improvements over the baseline Paper 2 run:
  1. 500 trajectories per scenario (D and E) instead of 100.
  2. Adaptive OLS R²-gate: physics prior zeroed when OLS R² < 0.7.
  3. Larger model: hidden_size=128, num_layers=2.
  4. Lower learning rate: 5e-4 (more stable for larger model).
  5. More epochs: 150 (more budget for complex dynamics).

Usage
-----
    python -m experiments.run_paper2_improved          # full (~30+ min)
    python -m experiments.run_paper2_improved --fast   # smoke (~2 min)
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from experiments import common
from experiments import e6_2qubit_improved


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E6 (ablation, noise sweep, cross-system on D and E)."""
    args = common.parse_fast(__doc__, argv)
    start = time.time()
    print(f"\n{'=' * 70}\nPAPER 2 IMPROVED — E6: Ablation + Noise + "
          f"Cross-system (D, E)\n{'=' * 70}")
    e6_2qubit_improved.main(["--fast"] if args.fast else [])
    print(f"\nPaper 2 improved experiments completed in "
          f"{(time.time() - start) / 60:.1f} min.")
    print("Results: experiments/results/e6_*_improved.csv")


if __name__ == "__main__":
    main()
