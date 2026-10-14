"""R8 — replicate the two Paper 2 claims that never went through a seed check.

Why
---
Five of Paper 2's conclusions were overturned by replication; two were
never replicated at all, and both rest on single runs:

  Claim 2  scaling and improved training push TFIM cross-system R^2 from
           -0.381 (pilot) to +0.575 (E8c).
  Claim 6  noise robustness is exceptional -- R^2 spans a total range of
           only 0.007 across sigma = 0 to 0.10 (E6b).

Claim 6 is the one to worry about. It is a statement that a spread is
*small*, made from one run, in a pipeline whose seed-to-seed spread
reaches +/-0.36. If the 0.007 range is an accident of a single
initialisation it says nothing about robustness, exactly as the
"R^2 traded for AUROC" reading turned out to say nothing.

Claim 2's direction is not in doubt -- the gap is +0.96, the same order
as Paper 1's central claim -- but the champion value 0.575 is quoted
throughout both papers, so it is worth knowing its spread.

Arms (three seeds each):

    e6b   noise sweep, sigma in {0, 0.01, 0.02, 0.05, 0.10}
    e8c   cross-system physics_lstm, trained on D+E, evaluated per scenario

Both call the published functions rather than copying their configuration.

Writes ONLY to experiments/results/r8_noise_and_scaling[_<tag>][_fast].csv.

Usage
-----
    python -m experiments.r8_noise_and_scaling --fast
    python -m experiments.r8_noise_and_scaling --seeds 42 43 44
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence

from experiments import common
from experiments import e6_2qubit_improved
from experiments import e8_improved_2qubit
from src.data import scenarios as scenario_lib
from src.evaluation import stats
from src.simulation import simulator as simulator_lib

COLUMNS = ("seed", "arm", "key", "r2", "mae", "auroc")

# Published R^2 from e6_noise_sweep_improved.csv (an earlier version of this
# file mistakenly held the AUROC column here).
PUBLISHED_E6B = {
    0.0: 0.5372, 0.01: 0.5383, 0.02: 0.5393, 0.05: 0.5440, 0.10: 0.5440}
PUBLISHED_E8C = {"D": 0.740, "E": 0.575}


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard deviation of the finite values."""
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan")
    return stats.mean_std(good)


def _run_arm(arm, simulator, groups, epochs, seed):
    """Runs one arm through the experiment function it replicates.

    Returns:
        Rows of (key, result) where key labels the cell of the table.
    """
    if arm == "e6b":
        rows = e6_2qubit_improved.run_e6b(
            simulator, groups, ["D", "E"], epochs, seed=seed, verbose=False)
        return [(f"{float(r['noise_sigma']):.2f}", r) for r in rows]
    rows = e8_improved_2qubit.run_e8c(
        simulator, groups, epochs, verbose=False, seed=seed)
    return [(str(r["name"]), r) for r in rows]


def _summarise(arms: Sequence[str],
               collected: Sequence[dict[str, object]]) -> None:
    """Prints mean R2 per cell and, for E6b, the span across sigma."""
    cells = defaultdict(list)
    for row in collected:
        cells[(row["arm"], row["key"])].append(float(row["r2"]))

    print("\n" + "=" * 62)
    for arm in arms:
        keys = sorted(k for (a, k) in cells if a == arm)
        if not keys:
            continue
        print(f"\n--- {arm} ---")
        print(f"{'key':<22}{'R2 mean':>10}{'±std':>9}{'published':>11}")
        for key in keys:
            mean, std = _mean_std(cells[(arm, key)])
            if arm == "e6b":
                published = PUBLISHED_E6B.get(float(key), float("nan"))
            else:
                published = next(
                    (v for k, v in PUBLISHED_E8C.items() if k in key),
                    float("nan"))
            print(f"{key:<22}{mean:>10.3f}{std:>9.3f}{published:>11.3f}")

        if arm == "e6b":
            # The claim is about the SPAN across sigma, within a run.
            per_seed = defaultdict(dict)
            for row in collected:
                if row["arm"] == "e6b":
                    per_seed[row["seed"]][row["key"]] = float(row["r2"])
            spans = [max(d.values()) - min(d.values())
                     for d in per_seed.values()]
            print(f"\n  span of R2 across sigma, per seed: "
                  f"{[f'{s:.3f}' for s in spans]}")
            mean_span = stats.mean_std(spans)[0]
            print(f"  mean span {mean_span:.3f} (published 0.007)")
            seed_sd = stats.mean_std(
                [_mean_std(cells[("e6b", k)])[1] for k in keys])[0]
            print(f"  typical seed-to-seed s.d. at fixed sigma: "
                  f"{seed_sd:.3f}")
            verdict = (
                "robustness claim holds: the sweep moves far less than "
                "the seeds do" if mean_span < seed_sd else
                "NOT supported: the sweep moves as much as the seeds do")
            print(f"  -> {verdict}")


def _add_flags(parser) -> None:
    parser.add_argument("--arms", nargs="+", default=["e6b", "e8c"])


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the arms for every seed and writes the CSV."""
    args = common.parse_args(__doc__, extra=_add_flags, argv=argv)
    n_per, epochs = (50, 10) if args.fast else (500, 150)
    path = common.results_path("r8_noise_and_scaling", args)

    print(f"R8 | seeds={args.seeds} arms={args.arms} n_per={n_per} "
          f"epochs={epochs}")
    print(f"writing -> {path}  (champion CSVs untouched)\n", flush=True)

    simulator = simulator_lib.QuTipSimulator()
    collected = []
    start = time.time()
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config={"arms": args.arms, "n_per": n_per,
                    "epochs": epochs}) as out:
        for seed in args.seeds:
            groups = scenario_lib.build_configs(
                n_per_scenario=n_per, seed=seed)
            for arm in args.arms:
                arm_start = time.time()
                print(f"--- seed {seed} | {arm} ---", flush=True)
                for key, result in _run_arm(
                        arm, simulator, groups, epochs, seed):
                    row = {"seed": seed, "arm": arm, "key": key,
                           "r2": result["r2"], "mae": result["mae"],
                           "auroc": result["auroc"]}
                    out.write(row)
                    collected.append(row)
                print(f"    [{time.time() - arm_start:.0f}s]", flush=True)

    _summarise(args.arms, collected)
    print(f"\ntotal {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
