"""Paper 1 — Single-qubit early warning (scenarios A, B, C).

Runs all experiments relevant to Paper 1:
  E1 — ablation on scenarios A (const γ), B (σ₋ time-dep), C (σ_z time-dep)
  E2 — minimum observation window sweep on B+C
  E3 — noise robustness on B+C
  E4 — cross-system generalisation, trained on A+B+C, evaluated per scenario
  E5 — direct vs inverse-problem baseline on B

Results land in experiments/results/ with suffix _p1 to avoid collisions with
global runs.

Usage::
    python experiments/run_paper1.py [--fast]

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
    import experiments.e2_window_sweep as e2
    import experiments.e3_noise_sweep as e3
    import experiments.e4_cross_system as e4
    import experiments.e5_inverse_baseline as e5

    n = 30 if args.fast else 100

    t_start = time.time()

    _log("PAPER 1 — E1: Ablation (scenarios A, B, C)")
    e1.main(scenarios=["A", "B", "C"], n_per=n, suffix="_p1")

    _log("PAPER 1 — E2: Window fraction sweep (scenarios B, C)")
    e2.main(scenarios=["B", "C"], suffix="_p1")

    _log("PAPER 1 — E3: Noise robustness (scenarios B, C)")
    e3.main(scenarios=["B", "C"], suffix="_p1")

    _log("PAPER 1 — E4: Cross-system generalisation (train A+B+C, eval A/B/C)")
    e4.main(
        train_scenarios=["A", "B", "C"],
        eval_scenarios=["A", "B", "C"],
        suffix="_p1",
    )

    _log("PAPER 1 — E5: Direct vs inverse-problem baseline (scenario B)")
    e5.main()

    elapsed = time.time() - t_start
    print(f"\nPaper 1 experiments completed in {elapsed/60:.1f} min.")
    print("Results in experiments/results/*_p1.csv (E5 → e5_inverse_baseline.csv)")


if __name__ == "__main__":
    main()
