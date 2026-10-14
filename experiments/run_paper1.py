"""Paper 1 — Single-qubit early warning (scenarios A, B, C).

Runs all experiments relevant to Paper 1:
  E1 — ablation on scenarios A (const γ), B (σ₋ time-dep), C (σ_z time-dep)
  E2 — minimum observation window sweep on B+C
  E3 — noise robustness on B+C
  E4 — cross-system generalisation, trained on A+B+C, evaluated per scenario
  E5 — direct vs inverse-problem baseline on B

Results land in experiments/results/ with the tag p1 (E5: no tag) to avoid
collisions with global runs. With --fast every experiment runs on reduced
data and writes `*_fast.csv`, which does not touch the published files.

Usage
-----
    python -m experiments.run_paper1 [--fast]
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
    """Runs E1-E5 for Paper 1."""
    args = common.parse_fast(__doc__, argv)
    fast = ["--fast"] if args.fast else []
    tag = ["--tag", "p1"]
    start = time.time()

    _log("PAPER 1 — E1: Ablation (scenarios A, B, C)")
    e1_ablation.main(
        ["--scenarios", "A", "B", "C", "--n-per", "100", *tag, *fast])

    _log("PAPER 1 — E2: Window fraction sweep (scenarios B, C)")
    e2_window_sweep.main(["--scenarios", "B", "C", *tag, *fast])

    _log("PAPER 1 — E3: Noise robustness (scenarios B, C)")
    e3_noise_sweep.main(["--scenarios", "B", "C", *tag, *fast])

    _log("PAPER 1 — E4: Cross-system generalisation (train A+B+C)")
    e4_cross_system.main([
        "--train-scenarios", "A", "B", "C",
        "--eval-scenarios", "A", "B", "C", *tag, *fast])

    _log("PAPER 1 — E5: Direct vs inverse-problem baseline (scenario B)")
    e5_inverse_baseline.main(fast)

    print(f"\nPaper 1 experiments completed in "
          f"{(time.time() - start) / 60:.1f} min.")
    print("Results in experiments/results/*_p1.csv "
          "(E5 → e5_inverse_baseline.csv)")


if __name__ == "__main__":
    main()
