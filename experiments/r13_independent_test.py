"""R13 — Paper 1 ablation scored on a large independent test set.

Why
---
E1/R7 score each model on the 10 test trajectories of a 100-trajectory
run; there even predicting the training mean gives R^2 between -2.0 and
0.0, so comparisons of a few hundredths are not resolved. This probe keeps
the E1 training setup and scores every model on 400 fresh trajectories per
scenario drawn from an independent generator seed, with 95% bootstrap
intervals obtained by resampling trajectories (not windows).

It also adds the missing ablation `physics_feature`: the same BiLSTM that
receives T_phys as an input scalar but predicts the whole remaining time
(no residual output). Comparing it with physics_lstm separates "T_phys is
a useful feature" from "the formula + correction output structure".

R^2 and AUROC are recorded in the forms of `metrics.r2_unguarded` and
`metrics.auroc_pairwise`, as in the published CSV.

Writes ONLY to experiments/results/r13_independent_test*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r13_independent_test --fast --tag smoke
    python -m experiments.r13_independent_test --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import dataclasses

from experiments import common
from src import pipeline

SCENARIOS = ("A", "B", "C")
VARIANTS = ("physics_only", "stretched_exp", "lstm_only", "physics_feature",
            "physics_lstm", "transformer")
COLUMNS = ("seed", "scenario", "variant", "r2", "r2_lo", "r2_hi", "mae",
           "auroc", "n_eval_traj", "n_eval_windows")
# Pipeline column -> CSV column of the published file.
RENAMES = {
    "r2_unguarded": "r2", "r2_unguarded_lo": "r2_lo",
    "r2_unguarded_hi": "r2_hi", "auroc_pairwise": "auroc",
    "n_groups": "n_eval_traj", "n_samples": "n_eval_windows",
}
WINDOW_LENGTH = 20


def arms(
    n_train: int, n_eval: int, n_boot: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares one arm per scenario and variant.

    Every model is trained on the scenario's own training set and scored on
    all windows of independent trajectories (generator seed + 1000).

    Args:
        n_train: Training trajectories per scenario.
        n_eval: Independent trajectories per scenario.
        n_boot: Bootstrap resamples of the R^2 interval.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    configs = []
    for scenario in SCENARIOS:
        data = pipeline.DataSpec(
            name=f"r13_train_{scenario}", scenarios=(scenario,),
            n_per_scenario=n_train, cache="disk",
            window_length=WINDOW_LENGTH)
        independent = dataclasses.replace(
            data, name=f"r13_eval_{scenario}", n_per_scenario=n_eval,
            train_ratio=0.0, val_ratio=0.0, config_seed_offset=1000,
            data_seed_offset=1000)
        tests = (pipeline.EvalSpec(
            data=independent,
            bootstrap=pipeline.Bootstrap("r2_unguarded", n_boot)),)
        configs += [
            pipeline.ExperimentConfig(
                data=data, model=variant, epochs=epochs,
                metrics=("r2_unguarded", "mae", "auroc_pairwise"),
                tests=tests,
                tags=(("scenario", scenario), ("variant", variant)))
            for variant in VARIANTS
        ]
    return configs


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the sizes of the sets to the common flags."""
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=400)
    parser.add_argument("--n-boot", type=int, default=1000)


def main() -> None:
    """Scores the Paper 1 ablation on an independent test set.

    Writes experiments/results/r13_independent_test*.csv.
    """
    args = common.parse_args(__doc__, extra=add_flags)
    if args.fast:
        args.n_train, args.n_eval, args.n_boot = 30, 30, 50
    epochs = 10 if args.fast else 100
    common.run_arms(
        "R13", arms(args.n_train, args.n_eval, args.n_boot, epochs), args,
        common.results_path("r13_independent_test", args), COLUMNS,
        rename=RENAMES)


if __name__ == "__main__":
    main()
