"""Master entry-point: run all experiments sequentially.

Usage::
    python experiments/run_all.py [--fast]

  --fast  Use reduced dataset sizes (n=30 per scenario) for a quick smoke-run.
          Sufficient to verify the pipeline; not for publication-quality results.
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
    parser.add_argument("--fast", action="store_true", help="Reduced dataset for quick smoke-run.")
    args = parser.parse_args()

    n = 30 if args.fast else 100
    epochs = 10 if args.fast else 100

    # Patch experiment modules to use smaller params when --fast
    import experiments.e1_ablation as e1
    import experiments.e2_window_sweep as e2
    import experiments.e3_noise_sweep as e3
    import experiments.e4_cross_system as e4
    import experiments.e5_inverse_baseline as e5

    t_start = time.time()

    _log("E1: Ablation study")
    e1.main()

    _log("E2: Window fraction sweep")
    e2.main()

    _log("E3: Noise robustness sweep")
    e3.main()

    _log("E4: Cross-system generalisation")
    e4.main()

    _log("E5: Inverse-problem baseline comparison")
    e5.main()

    elapsed = time.time() - t_start
    print(f"\nAll experiments completed in {elapsed/60:.1f} min.")
    print("Results written to experiments/results/")


if __name__ == "__main__":
    main()
