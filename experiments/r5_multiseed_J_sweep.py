"""R5 — is the predictability dip near the TFIM transition reproducible?

Why
---
Paper 2 reports a pronounced dip in prediction accuracy around the TFIM
quantum phase transition (worst at J=1.2, R^2=0.319, against 0.876 at
J=0.2 and 0.836 at J=1.5) and argues it is structural: the
amplitude-damping dissipator does not commute with the parity operator,
so it mixes the two symmetry sectors, most damagingly where their
timescales are comparable.

The argument is worked out in detail, but its entire empirical basis is
one sweep of seven J values at a single seed, with 81-94 evaluation
samples per point. Everywhere else in this pipeline that a single run
looked decisive, replication has either reversed it or dissolved it, so
the dip needs the same treatment before the structural reading can be
relied on.

What this measures
------------------
The same sweep across seeds, varying both the sampled configurations
(config_seed) and training (seed). This answers whether the dip is
there at all and whether its location is stable; it does not test the
proposed mechanism, which would need a dissipator that commutes with
parity.

Writes ONLY to experiments/results/r5_multiseed_J_sweep*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r5_multiseed_J_sweep --fast --tag smoke
    python -m experiments.r5_multiseed_J_sweep --seeds 42 43 44
"""
from __future__ import annotations

import collections
import math
from collections.abc import Mapping, Sequence

from experiments import common
from experiments import j_sweep
from src.evaluation import stats

COLUMNS = ("seed", "J", "r2", "mae", "auroc", "n_samples")
# R^2 of the single-seed sweep in paper 2, by J.
PUBLISHED_R2 = {0.2: 0.876, 0.5: 0.770, 0.8: 0.469, 1.0: 0.497,
                1.2: 0.319, 1.5: 0.836, 2.0: 0.614}


def finite_mean_std(values: Sequence[float]) -> tuple[float, float, int]:
    """Mean and s.d. over the finite entries, and how many there are.

    AUROC is nan when a bucket has only one risk class, and a degenerate
    point must not kill the summary.
    """
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan"), 0
    mean, std = stats.mean_std(good)
    return mean, std, len(good)


def summarize(rows: Sequence[Mapping[str, object]]) -> None:
    """Prints R^2 and AUROC by J, the dip depth and the worst J per seed.

    Args:
        rows: Result rows with `seed`, `J`, `r2` and `auroc`.
    """
    by_j = collections.defaultdict(list)
    for row in rows:
        by_j[float(row["J"])].append(row)

    print("\n" + "=" * 66)
    print(f"{'J':>5}{'r(J)':>7}{'R2 mean':>10}{'±std':>8}{'AUROC':>8}"
          f"{'±std':>7}{'published':>11}")
    for coupling in sorted(by_j):
        mean_r2, std_r2, _ = finite_mean_std(
            [float(r["r2"]) for r in by_j[coupling]])
        mean_au, std_au, n_au = finite_mean_std(
            [float(r["auroc"]) for r in by_j[coupling]])
        flag = "" if n_au == len(by_j[coupling]) else f"  (AUROC n={n_au})"
        published = PUBLISHED_R2.get(coupling, float("nan"))
        print(f"{coupling:>5.1f}{j_sweep.splitting_ratio(coupling):>7.3f}"
              f"{mean_r2:>10.3f}{std_r2:>8.3f}{mean_au:>8.3f}{std_au:>7.3f}"
              f"{published:>11.3f}{flag}")

    spread_stats = {
        coupling: finite_mean_std([float(r["r2"]) for r in group])
        for coupling, group in by_j.items()}
    means = {c: m for c, (m, _, _) in spread_stats.items()
             if math.isfinite(m)}
    if not means:
        return
    worst, best = min(means, key=means.get), max(means, key=means.get)
    spread = means[best] - means[worst]
    stds = [s for c, (_, s, _) in spread_stats.items()
            if len(by_j[c]) > 1 and math.isfinite(s)]
    typical = stats.mean_std(stds)[0] if stds else 0.0
    print(f"\nworst mean at J={worst} ({means[worst]:.3f}), "
          f"best at J={best} ({means[best]:.3f})")
    print(f"dip depth {spread:.3f} vs typical seed spread {typical:.3f}"
          f"  -> {'resolved' if spread > 2 * typical else 'NOT resolved'}")
    worst_per_seed = {}
    for row in rows:
        r2 = float(row["r2"])
        seed = row["seed"]
        if math.isfinite(r2) and (
                seed not in worst_per_seed
                or r2 < float(worst_per_seed[seed]["r2"])):
            worst_per_seed[seed] = row
    print("worst J per seed: " + ", ".join(
        f"seed {seed}: J={float(row['J']):.1f}"
        for seed, row in sorted(worst_per_seed.items())))


def main() -> None:
    """Runs the multi-seed TFIM J sweep and prints the per-J summary.

    Writes experiments/results/r5_multiseed_J_sweep*.csv.
    """
    args = common.parse_args(__doc__)
    n_per_j, epochs = (30, 10) if args.fast else (200, 150)
    rows = common.run_arms(
        "R5", j_sweep.arms("r5", n_per_j, epochs), args,
        common.results_path("r5_multiseed_J_sweep", args), COLUMNS)
    summarize(rows)


if __name__ == "__main__":
    main()
