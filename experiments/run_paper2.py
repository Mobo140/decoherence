"""Paper 2 — Two-qubit early warning (scenarios D, E).

Runs all experiments relevant to Paper 2:
  E1 — ablation on scenarios D (XXZ time-dep γ) and E (TFIM time-dep γ)
  E3 — noise robustness on D+E
  E4 — cross-system generalisation, trained on D+E, evaluated per scenario

Note: E2 (window sweep) and E5 (inverse baseline) are omitted for Paper 2.
  E2 can be added later; E5 is scoped to 1-qubit in Paper 1.

Results land in experiments/results/ with suffix _p2.

Usage::
    python experiments/run_paper2.py [--fast]

  --fast  Use n=30 per scenario and 10 epochs for a smoke-run.
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
    parser.add_argument("--fast", action="store_true", help="Smoke-run with reduced data.")
    args = parser.parse_args()

    import experiments.e1_ablation as e1
    import experiments.e3_noise_sweep as e3
    import experiments.e4_cross_system as e4

    n = 30 if args.fast else 100

    t_start = time.time()

    _log("PAPER 2 — E1: Ablation (scenarios D, E)")
    e1.main(scenarios=["D", "E"], n_per=n, suffix="_p2")

    _log("PAPER 2 — E3: Noise robustness (scenarios D, E)")
    e3.main(scenarios=["D", "E"], suffix="_p2")

    _log("PAPER 2 — E4: Cross-system generalisation (train D+E, eval D/E)")
    e4.main(
        train_scenarios=["D", "E"],
        eval_scenarios=["D", "E"],
        suffix="_p2",
    )

    elapsed = time.time() - t_start
    print(f"\nPaper 2 experiments completed in {elapsed/60:.1f} min.")
    print("Results in experiments/results/*_p2.csv")


if __name__ == "__main__":
    main()
