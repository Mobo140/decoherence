"""R7 — multi-seed replication of Paper 1's ablation (E1, scenarios A-C).

Why
---
Paper 1 was never put through the check that dismantled five of Paper 2's
conclusions. Its numbers are single runs too.

Sorting its claims by margin shows where the risk actually sits. The
central one is not at risk: lstm_only scores -0.446, 0.009 and -0.029
against roughly 0.95 for physics_lstm, so "the physics prior is
essential" has a margin of order 1.0 in R^2. Neither is C1
(physics_only = 0.9995 on the constant-gamma scenario, an analytic
result), nor physics_lstm over physics_only on B (+0.19), nor the
residual over the Transformer on A (+0.18).

Two comparisons do rest on thin margins, both on scenario C:

    physics_lstm 0.9756 vs physics_only 0.9649   -> +0.011
    transformer  0.9808 vs physics_lstm  0.9756  -> +0.005

For scale, replication dissolved margins of 0.037 and 0.04 in Paper 2.

This probe calls e1_ablation.run rather than rebuilding the command, so
it cannot drift from the experiment it replicates -- E1 uses the base
AblationStudyUseCase, whose default predictors are what Paper 1 reports,
so unlike R1 there is no override to mirror.

Writes ONLY to experiments/results/r7_multiseed_1qubit[_<tag>][_fast].csv
(R11 is this script with --n-per 500 --out r11_1qubit_500traj.csv).

Usage
-----
    python -m experiments.r7_multiseed_1qubit --fast
    python -m experiments.r7_multiseed_1qubit --seeds 42 43 44
"""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence

from experiments import common
from experiments import e1_ablation
from src.evaluation import stats

COLUMNS = ("seed", "scenario", "variant", "r2", "mae", "auroc")
SCENARIOS = ("A", "B", "C")

# Published single-run values (e1_ablation_p1.csv), for the comparison column.
PUBLISHED = {
    ("A", "physics_only"): 0.9995, ("A", "physics_lstm"): 0.9756,
    ("A", "lstm_only"): -0.4462,   ("A", "transformer"): 0.7991,
    ("B", "physics_only"): 0.7534, ("B", "physics_lstm"): 0.9449,
    ("B", "lstm_only"): 0.0094,    ("B", "transformer"): 0.9024,
    ("C", "physics_only"): 0.9649, ("C", "physics_lstm"): 0.9756,
    ("C", "lstm_only"): -0.0292,   ("C", "transformer"): 0.9808,
}

# The claims worth testing, as (scenario, better, worse, published margin).
CLAIMS = [
    ("A", "physics_only", "physics_lstm", 0.0239,
     "C1: formula exact at const gamma"),
    ("B", "physics_lstm", "physics_only", 0.1915,
     "C2: prior + residual beats formula (B)"),
    ("C", "physics_lstm", "physics_only", 0.0107,
     "C2: prior + residual beats formula (C)"),
    ("A", "physics_lstm", "lstm_only", 1.4218, "C2: prior is essential (A)"),
    ("B", "physics_lstm", "lstm_only", 0.9355, "C2: prior is essential (B)"),
    ("C", "physics_lstm", "lstm_only", 1.0048, "C2: prior is essential (C)"),
    ("A", "physics_lstm", "transformer", 0.1765,
     "residual preferred at const gamma"),
    ("C", "transformer", "physics_lstm", 0.0052, "Transformer edge on C"),
]


def _add_flags(parser) -> None:
    parser.add_argument("--n-per", type=int, default=100,
                        help="trajectories per scenario (R11 uses 500)")


def _summarise(collected: Sequence[dict[str, object]]) -> None:
    """Prints mean R2 per cell and the paired claims over the seeds."""
    by_cell = defaultdict(dict)
    for row in collected:
        by_cell[(row["scenario"], row["variant"])][row["seed"]] = float(
            row["r2"])

    print("\n" + "=" * 66)
    print(f"{'scen':<5}{'variant':<15}{'R2 mean':>10}{'±std':>8}"
          f"{'published':>11}")
    for key in sorted(by_cell):
        mean, std = stats.mean_std(list(by_cell[key].values()))
        published = PUBLISHED.get(key, float("nan"))
        print(f"{key[0]:<5}{key[1]:<15}{mean:>10.3f}{std:>8.3f}"
              f"{published:>11.4f}")

    print("\nclaims, as paired differences per seed:")
    for scenario, better, worse, published, label in CLAIMS:
        high = by_cell.get((scenario, better), {})
        low = by_cell.get((scenario, worse), {})
        seeds = sorted(set(high) & set(low))
        if not seeds:
            continue
        diff = stats.paired_difference(
            [high[s] for s in seeds], [low[s] for s in seeds])
        if diff.n_positive == diff.n:
            verdict = "holds in every seed"
        elif diff.n_negative == diff.n:
            verdict = "REVERSED in every seed"
        else:
            verdict = "SIGN VARIES -- not resolved"
        print(f"  {label}")
        print(f"    published {published:+.4f} | replicated "
              f"{diff.mean:+.3f} ± {diff.std:.3f}  "
              f"{[f'{x:+.3f}' for x in diff.differences]}  -> {verdict}")


def main(argv: Sequence[str] | None = None) -> None:
    """Replicates E1 on scenarios A-C for each seed."""
    args = common.parse_args(__doc__, extra=_add_flags, argv=argv)
    n_per = 30 if args.fast else args.n_per
    path = common.results_path("r7_multiseed_1qubit", args)

    print(f"R7 | seeds={args.seeds} n_per={n_per} scenarios=A,B,C")
    print(f"writing -> {path}  (champion CSVs untouched)\n", flush=True)

    start = time.time()
    collected = []
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config={"n_per": n_per, "scenarios": SCENARIOS}) as out:
        for seed in args.seeds:
            seed_start = time.time()
            print(f"--- seed {seed} ---", flush=True)
            for result in e1_ablation.run(SCENARIOS, n_per, seed=seed):
                row = {"seed": seed, **result}
                out.write(row)
                collected.append(row)
            print(f"    [{time.time() - seed_start:.0f}s]", flush=True)

    _summarise(collected)
    print(f"\ntotal {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
