"""Paper 2 improved — Two-qubit early warning with roadmap fixes applied.

Improvements over the baseline Paper 2 run:
  1. 500 trajectories per scenario (D and E) instead of 100.
  2. Adaptive OLS R²-gate: physics prior zeroed when OLS R² < 0.7.
  3. Larger model: hidden_size=128, num_layers=2.
  4. Lower learning rate: 5e-4 (more stable for larger model).
  5. More epochs: 150 (more budget for complex dynamics).

Run::
    python experiments/run_paper2_improved.py          # full run (~30+ min)
    python experiments/run_paper2_improved.py --fast   # smoke-run (~2 min)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _log(msg: str) -> None:
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Smoke-run with n=50, epochs=10.")
    args = parser.parse_args()

    import experiments.e6_2qubit_improved as e6

    t_start = time.time()

    _log("PAPER 2 IMPROVED — E6: Ablation + Noise + Cross-system (D, E)")
    e6.main(fast=args.fast)

    elapsed = time.time() - t_start
    print(f"\nPaper 2 improved experiments completed in {elapsed/60:.1f} min.")
    print("Results: experiments/results/e6_*_improved.csv")


if __name__ == "__main__":
    main()
