"""R4 — does E11a's score depend on evaluating the training configurations?

Why
---
E11a trains on ``build_configs(seed=42)`` and then evaluates on the *same*
configurations, regenerating their trajectories under a different seed.
The model therefore sees fresh noise realisations of physical systems it
has already been trained on, not unseen systems.

Whether that inflates the score depends on how much of T2 the
configuration fixes. Measured directly, the spread of T2 within one
configuration is much smaller than the spread across configurations on
XXZ (roughly 0.25--0.53 against 1.90) and comparable on TFIM
(0.73--1.50 against 1.46). So on XXZ a model that recognised a
configuration from its window could recover most of the target without
generalising at all.

Design
------
One trained model per seed, scored twice:

    seen     the training configurations, fresh trajectories (published)
    unseen   disjoint configurations from a different build_configs seed

Everything else -- architecture, window, min-gap, protocol, evaluation
seed -- is identical, so the difference is attributable to configuration
overlap alone. A large gap would mean the published cross-system numbers
are optimistic; a small one closes the question.

Writes ONLY to experiments/results/r4_config_overlap[_<tag>][_fast].csv.

Usage
-----
    python -m experiments.r4_config_overlap --fast
    python -m experiments.r4_config_overlap --seeds 42 43 44
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
CONFIG_SEED_OFFSET = 500      # far from any seed the experiments use


def _summarise(collected: Sequence[dict[str, object]]) -> None:
    """Prints R2 and AUROC per arm and the paired seen - unseen gaps."""
    cells = defaultdict(dict)
    for row in collected:
        cells[(row["scenario"], row["arm"])][row["seed"]] = row

    print("\n" + "=" * 58)
    print(f"{'scen':<5}{'arm':<8}{'R2 mean':>9}{'±std':>8}{'AUROC':>8}"
          f"{'±std':>7}")
    for scenario in ("D", "E"):
        for arm in ("seen", "unseen"):
            values = list(cells[(scenario, arm)].values())
            if not values:
                continue
            mean_r2, std_r2 = stats.mean_std(
                [float(x["r2"]) for x in values])
            mean_au, std_au = stats.mean_std(
                [float(x["auroc"]) for x in values])
            print(f"{scenario:<5}{arm:<8}{mean_r2:9.3f}{std_r2:8.3f}"
                  f"{mean_au:8.3f}{std_au:7.3f}")

    print("\npaired seen - unseen (positive = evaluating on training "
          "configs flatters):")
    for scenario in ("D", "E"):
        seen_cells = cells[(scenario, "seen")]
        unseen_cells = cells[(scenario, "unseen")]
        seeds = sorted(set(seen_cells) & set(unseen_cells))
        if not seeds:
            continue
        seen = [seen_cells[s] for s in seeds]
        unseen = [unseen_cells[s] for s in seeds]
        d_r2 = stats.paired_difference(
            [float(x["r2"]) for x in seen], [float(x["r2"]) for x in unseen])
        d_au = stats.paired_difference(
            [float(x["auroc"]) for x in seen],
            [float(x["auroc"]) for x in unseen])
        print(f"  {scenario}: ΔR²={d_r2.mean:+.3f} ± {d_r2.std:.3f}  "
              f"{[f'{x:+.3f}' for x in d_r2.differences]}")
        print(f"     ΔAUROC={d_au.mean:+.3f}  "
              f"{[f'{x:+.3f}' for x in d_au.differences]}")


def main(argv: Sequence[str] | None = None) -> None:
    """Scores one model per seed on seen and on unseen configurations."""
    args = common.parse_args(__doc__, argv=argv)
    n_per, epochs = (50, 10) if args.fast else (500, 150)
    path = common.results_path("r4_config_overlap", args)

    print(f"R4 | seeds={args.seeds} n_per={n_per} epochs={epochs}")
    print(f"writing -> {path}  (champion CSVs untouched)\n", flush=True)

    simulator = simulator_lib.QuTipSimulator()
    collected = []
    start = time.time()
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config={"n_per": n_per, "epochs": epochs,
                    "config_seed_offset": CONFIG_SEED_OFFSET}) as out:
        for seed in args.seeds:
            train_groups = scenario_lib.build_configs(
                n_per_scenario=n_per, seed=seed)
            fresh_groups = scenario_lib.build_configs(
                n_per_scenario=n_per, seed=seed + CONFIG_SEED_OFFSET)
            for arm, eval_groups in (("seen", None),
                                     ("unseen", fresh_groups)):
                arm_start = time.time()
                print(f"--- seed {seed} | {arm} configs ---", flush=True)
                rows = e11_scaling.run_e11a(
                    simulator, train_groups, epochs, n_per, verbose=False,
                    seed=seed, window=20, eval_groups=eval_groups)
                for result in rows:
                    row = {"seed": seed, "arm": arm,
                           "scenario": result["scenario"],
                           "r2": result["r2"], "mae": result["mae"],
                           "auroc": result["auroc"],
                           "n_samples": result["n_samples"]}
                    out.write(row)
                    collected.append(row)
                    print(f"    {arm:<7} {result['scenario']} "
                          f"R2={result['r2']:8.3f} "
                          f"AUROC={result['auroc']:.3f}", flush=True)
                print(f"    [{time.time() - arm_start:.0f}s]", flush=True)

    _summarise(collected)
    print(f"\ntotal {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
