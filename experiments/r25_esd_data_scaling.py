"""R25 — learning curve for the entanglement-sudden-death target.

Why
---
R24 found T_ESD (first zero of the Wootters concurrence) predictable at
R^2 = 0.43-0.66 with 500 trajectories, while a 2% change of the rate moves
it by only 0.1-0.3%: the first death is set mostly by the state and the
Hamiltonian, which a window with the full state contains. That suggests
the remaining error is a learning problem rather than an information
limit. On XXZ the concurrence as an explicit input helped by 0.08 over the
correlators it is computed from, i.e. the network did not learn the
nonlinear map from 500 trajectories. Two predictions, if that reading is
right: R^2 keeps rising with the training set, and the gap between
"+ correlators" and "+ correlators + concurrence" closes.

Setup: Transformer (as in E11a/R24), window 20, scenarios D and E,
training sets of 250, 500, 1000, 2000 and 4000 trajectories, three input
sets (base / corr / corr_conc, as in R24), three seeds. The training sets
are nested: the 4000 trajectories are simulated once and each smaller set
is the first N of them (an in-memory store hands the same trajectories
back), so the curve is not confounded by resampling. Every model is
scored on the same 200 independent trajectories per scenario and seed as
in R24 (generator seed + 1000). Censored trajectories (still entangled at
t_max) are dropped, as everywhere.

Writes ONLY to experiments/results/r25_esd_data_scaling*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r25_esd_data_scaling --fast --tag smoke
    python -m experiments.r25_esd_data_scaling --seeds 42 --scenario D --tag D42
"""
from __future__ import annotations

import argparse
import dataclasses

from experiments import common
from experiments.r23_two_body_correlators import BATCH_SIZE, LEARNING_RATE
from experiments.r24_entanglement_target import ESD
from src import pipeline

SIZES = (250, 500, 1000, 2000, 4000)
FEATURES = ("base", "corr", "corr_conc")
COLUMNS = ("seed", "scenario", "n_traj", "features", "r2", "mae", "auroc",
           "n_samples", "n_train_windows", "n_censored")


def arms(
    scenario: str, sizes: tuple[int, ...], n_eval: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the learning-curve arms of one scenario.

    The training sets are nested: the configurations of the largest set
    are drawn once and each smaller set is the first N of them. The arms
    run from the largest size down and keep their trajectories in memory,
    so the smaller sets are hits and the stream of initial states is never
    shifted. Every model is scored on the same independent trajectories
    (generator seed + 1000).

    Args:
        scenario: "D" or "E".
        sizes: Training-set sizes in trajectories.
        n_eval: Independent trajectories for scoring.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    pool = pipeline.DataSpec(
        name="r25", scenarios=(scenario,), n_per_scenario=max(sizes),
        criterion=ESD.value, cache="memory",
        simulator=pipeline.SimulatorSpec("entanglement"), window_length=20)
    independent = dataclasses.replace(
        pool, name="r25_eval", n_per_scenario=n_eval, cache="none",
        train_ratio=0.0, val_ratio=0.0, config_seed_offset=1000,
        data_seed_offset=1)
    tests = (pipeline.EvalSpec(data=independent),)
    return [
        pipeline.ExperimentConfig(
            data=dataclasses.replace(pool, n_trajectories=n),
            model="transformer_large", features=feats, epochs=epochs,
            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, tests=tests,
            tags=(("scenario", scenario), ("n_traj", n),
                  ("features", feats)))
        for n in sorted(sizes, reverse=True)
        for feats in FEATURES
    ]


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of scenarios to the common flags."""
    parser.add_argument("--scenario", choices=["D", "E", "DE"],
                        default="DE")


def main() -> None:
    args = common.parse_args(__doc__, extra=add_flags)
    sizes, n_eval, epochs = (
        ((20, 40), 20, 5) if args.fast else (SIZES, 200, 150))
    configs = [config for scenario in args.scenario
               for config in arms(scenario, sizes, n_eval, epochs)]
    common.run_arms(
        "R25", configs, args,
        common.results_path("r25_esd_data_scaling", args), COLUMNS)


if __name__ == "__main__":
    main()
