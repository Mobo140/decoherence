"""R10 — does the TFIM J-dependence disappear with an energy-basis target?

Why
---
R9 found that the T2 target, the first 1/e crossing of the l1-coherence in
the computational basis, is modulated by the TFIM Hamiltonian with a depth
set by tan(2 theta) = J / 2h, and that the share of trajectories whose
coherence re-crosses the threshold tracks the R5 loss in R^2 across J
(r = -0.85). That is a correlation over seven points.

This is the direct test: repeat R5 exactly (same configurations, seeds,
model, training) and change only the target, to the 1/e crossing of the
l1-coherence in the eigenbasis of H, which unitary evolution does not
modulate. The model inputs stay in the computational basis.

  * If R^2(J) flattens, the TFIM ceiling is a property of the target.
  * If the decline persists, its cause is in the dynamics.

The comparison is paired with R5 by seed (same config_seed, same seed).

Writes ONLY to experiments/results/r10_energy_target_J_sweep*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r10_energy_target_J_sweep --fast --tag smoke
    python -m experiments.r10_energy_target_J_sweep --seeds 42 43 44
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence

from experiments import common
from experiments import j_sweep
from src.domain import value_objects
from src.evaluation import stats

COLUMNS = ("seed", "J", "r2", "mae", "auroc", "n_samples")
R5 = common.RESULTS_DIR / "r5_multiseed_J_sweep.csv"
# J below which the coupling is weak, and from which it is strong.
WEAK_MAX, STRONG_MIN = 0.5, 0.8


def r2_by_coupling(
    rows: Sequence[Mapping[str, object]]
) -> dict[float, dict[int, float]]:
    """Groups R^2 by coupling and seed."""
    grouped: dict[float, dict[int, float]] = {}
    for row in rows:
        grouped.setdefault(float(row["J"]), {})[int(row["seed"])] = float(
            row["r2"])
    return grouped


def drop(r2: Mapping[float, Mapping[int, float]]) -> float:
    """Mean R^2 at weak coupling minus mean R^2 at strong coupling."""
    weak = [v for j, by_seed in r2.items() if j <= WEAK_MAX
            for v in by_seed.values()]
    strong = [v for j, by_seed in r2.items() if j >= STRONG_MIN
              for v in by_seed.values()]
    return stats.mean_std(weak)[0] - stats.mean_std(strong)[0]


def compare_with_r5(rows: Sequence[Mapping[str, object]]) -> None:
    """Prints R10 against the committed R5 sweep, paired by seed.

    Args:
        rows: R10 result rows with `seed`, `J` and `r2`.
    """
    with R5.open() as f:
        base = r2_by_coupling(list(csv.DictReader(f)))
    new = r2_by_coupling(rows)
    print(f"\n{'J':>5}{'R5 R2':>10}{'R10 R2':>10}{'paired diff':>14}")
    for coupling in sorted(new):
        seeds = sorted(set(new[coupling]) & set(base[coupling]))
        diff = stats.paired_difference(
            [new[coupling][s] for s in seeds],
            [base[coupling][s] for s in seeds])
        print(f"{coupling:>5.1f}"
              f"{stats.mean_std(list(base[coupling].values()))[0]:>10.3f}"
              f"{stats.mean_std(list(new[coupling].values()))[0]:>10.3f}"
              f"{diff.mean:>+14.3f}")
    print(f"\ndrop (J<={WEAK_MAX} minus J>={STRONG_MIN}): "
          f"R5 {drop(base):+.3f}  R10 {drop(new):+.3f}")


def main() -> None:
    args = common.parse_args(__doc__)
    n_per_j, epochs = (30, 10) if args.fast else (200, 150)
    arms = j_sweep.arms(
        "r10", n_per_j, epochs,
        criterion=value_objects.DecoherenceCriterion.COHERENCE_ENERGY.value)
    rows = common.run_arms(
        "R10", arms, args,
        common.results_path("r10_energy_target_J_sweep", args), COLUMNS)
    if not args.fast and R5.exists():
        compare_with_r5(rows)


if __name__ == "__main__":
    main()
