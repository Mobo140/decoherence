"""R3 — multi-seed check of the two E11 claims Paper 2 marks as hypotheses.

Why
---
Paper 2 carries two scaling statements that were never replicated:

  * "physics_lstm degrades as data grows" — E11b (1000 trajs) scores
    cross-system XXZ R²=0.491 against E8c's 0.740 (500 trajs).
  * "E11a trades R² for AUROC" — E11a scores R²=0.566 / AUROC=0.888 on
    TFIM against E7a's 0.599 / 0.736.

Neither comparison isolates the factor it names.  E8c and E11b differ in
window length (30 vs 20), early-stopping patience and inference codes as
well as in trajectory count; E7a and E11a differ in window length (50 vs
20), min-gap and the whole evaluation protocol.  A difference between two
such runs cannot be attributed to data size or to window length, and a
single run cannot separate either from seed noise.

Design
------
Four arms, three seeds each, every arm evaluated through the *same*
protocol (E11's ``_eval_per_scenario``), so arms differ in one factor:

    e11a_w20    Transformer, window 20, 500 trajs   (published E11a)
    e7a_w50     Transformer, window 50, 500 trajs   <- window contrast
    e11b_500    physics_lstm E11b config, 500 trajs <- data-size contrast
    e11b_1000   physics_lstm E11b config, 1000 trajs (published E11b)

The arms call ``run_e11a`` / ``run_e11b`` directly rather than copying
their configuration, so the probe cannot drift away from the experiment
it replicates. Those functions return rows and write nothing, so the
champion CSVs cannot be touched.  ``e7a_w50`` is *not* E7a: it applies
E7a's window inside E11a's protocol, and so isolates the window without
reproducing E7a's published number.

Writes ONLY to experiments/results/r3_multiseed_scaling[_<tag>][_fast].csv.

Usage
-----
    python -m experiments.r3_multiseed_scaling --fast
    python -m experiments.r3_multiseed_scaling --seeds 42 43 44
"""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence

from experiments import common
from experiments import e11_scaling
from src.data import scenarios as scenario_lib
from src.evaluation import stats
from src.simulation import simulator as simulator_lib

COLUMNS = ("seed", "arm", "scenario", "r2", "mae", "auroc", "n_samples")
ARMS = ("e11a_w20", "e7a_w50", "e11b_500", "e11b_1000")


def summarise(rows: Sequence[dict[str, object]]) -> None:
    """Prints mean and standard deviation of R^2 and AUROC per arm.

    Args:
        rows: Result rows with the keys arm, scenario, r2 and auroc.
    """
    print("\n" + "=" * 62)
    print(f"{'arm':<11}{'scen':<5}{'R2 mean':>9}{'R2 std':>9}{'AUROC':>8}"
          f"{'std':>7}{'n':>4}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["scenario"])].append(row)
    for key in sorted(groups):
        r2_mean, r2_std = stats.mean_std(
            [float(x["r2"]) for x in groups[key]])
        au_mean, au_std = stats.mean_std(
            [float(x["auroc"]) for x in groups[key]])
        print(f"{key[0]:<11}{key[1]:<5}{r2_mean:9.3f}{r2_std:9.3f}"
              f"{au_mean:8.3f}{au_std:7.3f}{len(groups[key]):4d}")


def _run_arm(arm, simulator, groups_500, groups_1000, epochs, n_500, seed):
    """Runs one arm through the experiment function it replicates."""
    if arm == "e11a_w20":
        return e11_scaling.run_e11a(
            simulator, groups_500, epochs, n_500, verbose=False, seed=seed,
            window=20)
    if arm == "e7a_w50":
        return e11_scaling.run_e11a(
            simulator, groups_500, epochs, n_500, verbose=False, seed=seed,
            window=50)
    if arm == "e11b_500":
        return e11_scaling.run_e11b(
            simulator, groups_500, epochs, verbose=False, seed=seed)
    if arm == "e11b_1000":
        return e11_scaling.run_e11b(
            simulator, groups_1000, epochs, verbose=False, seed=seed)
    raise ValueError(f"unknown arm: {arm}")


def _add_flags(parser) -> None:
    parser.add_argument("--arms", nargs="+", default=list(ARMS))


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the arms for every seed and writes the CSV."""
    args = common.parse_args(__doc__, extra=_add_flags, argv=argv)
    n_500, n_1000, epochs = (
        (50, 100, 10) if args.fast else (500, 1000, 150))
    path = common.results_path("r3_multiseed_scaling", args)

    print(f"R3 | seeds={args.seeds} arms={args.arms} n={n_500}/{n_1000} "
          f"epochs={epochs}")
    print(f"writing -> {path}  (champion CSVs untouched)\n", flush=True)

    simulator = simulator_lib.QuTipSimulator()
    collected = []
    start = time.time()
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config={"arms": args.arms, "n_500": n_500, "n_1000": n_1000,
                    "epochs": epochs}) as out:
        for seed in args.seeds:
            groups_500 = scenario_lib.build_configs(
                n_per_scenario=n_500, seed=seed)
            groups_1000 = scenario_lib.build_configs(
                n_per_scenario=n_1000, seed=seed)
            for arm in args.arms:
                arm_start = time.time()
                print(f"--- seed {seed} | {arm} ---", flush=True)
                rows = _run_arm(arm, simulator, groups_500, groups_1000,
                                epochs, n_500, seed)
                for result in rows:
                    row = {"seed": seed, "arm": arm,
                           "scenario": result["scenario"],
                           "r2": result["r2"], "mae": result["mae"],
                           "auroc": result["auroc"],
                           "n_samples": result["n_samples"]}
                    out.write(row)
                    collected.append(row)
                    print(f"    {arm:<10} {result['scenario']} "
                          f"R2={result['r2']:8.3f} "
                          f"AUROC={result['auroc']:.3f}", flush=True)
                print(f"    [{time.time() - arm_start:.0f}s]", flush=True)

    summarise(collected)
    print(f"\ntotal {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
