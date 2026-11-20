"""R30 — two qubits: which inputs, exact or from a finite number of shots.

Why
---
For two qubits purity and l1 coherence are global quantities: six local
Pauli means do not give them, and the base input set (6 means, purity,
coherence_l1) in effect needs full tomography. This experiment compares what
a model gets from different measurements of the same states.

Input sets (the same simulated states and windows in all of them):

  i    exact_base         6 local means, purity, coherence_l1, from rho(t)
  ii   exact_local        the 6 local means only, from rho(t)
  iii  local_1000         6 local means from 1000 shots per basis (3 bases);
                          no purity or coherence
  iv   tomography_N       6 means, 9 correlators, purity and coherence_l1 of
                          the linear-inversion estimate of rho (9 bases), for
                          N = 1000 (and 100, 10000 with --with-n-grid)

Scenarios D (XXZ) and E (TFIM), 500 trajectories each, window 20; targets T2
(computational basis) and T2^E (energy basis). A model is trained on the
"train" noise stream of its input set and scored on the "test" stream of the
same set; the split is 0.7 / 0.1 / 0.2 by trajectory and the test
trajectories are the same for every input set.

`physics_only` and `physics_lstm` read purity and coherence_l1 by position
(`observe.MODEL_TAIL`), so they run only on the sets that have them (i and
iv). `lstm_only` also computes its scalar inputs from the last two columns
(by name when `coherence_l1` and `purity` exist, otherwise the last two
columns, which on sets ii and iii are sigma_y2 and sigma_z2); this is
reported, not changed.

Writes ONLY to experiments/results/r30_finite_shots_2q*.csv with its
provenance file.

Usage
-----
    python -m experiments.new.r30_finite_shots_2q --fast --tag smoke
    python -m experiments.new.r30_finite_shots_2q --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Iterator, Sequence

from experiments import common
from experiments.new import shot_harness as harness
from src import pipeline
from src.data import shots
from src.domain import value_objects

SCENARIOS = ("D", "E")
# CSV label -> criterion of the target (None keeps the scenario's T2).
TARGETS = {
    "t2e": value_objects.DecoherenceCriterion.COHERENCE_ENERGY.value,
    "t2": None,
}
# Input sets of the plan: (label, kind, n_shots).
INPUT_SETS = (
    ("i", "exact", None),
    ("ii", "exact_local", None),
    ("iii", "local", 1000),
    ("iv", "tomography", 1000),
)
EXTRA_SETS = (
    ("v", "tomography", 100),
    ("v", "tomography", 10000),
)
MODELS = ("physics_only", "physics_lstm", "lstm_only", "transformer")
# Models that read the columns purity and coherence_l1 by position.
TAIL_MODELS = ("physics_only", "physics_lstm")
COLUMNS = ("seed", "scenario", "target", "input_set", "kind", "n_shots",
           "model", "n_features", "r2", "mae", "auroc", "n_samples",
           "train_seconds")
WINDOW_LENGTH = 20
HORIZON = 1.0
SAMPLES = 5
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
TRAIN_RATIO, VAL_RATIO = 0.7, 0.1


def data_spec(
    scenario: str, target: str, n_per: int
) -> pipeline.DataSpec:
    """Declares the data of one scenario and target.

    Args:
        scenario: "D" or "E".
        target: A key of `TARGETS`.
        n_per: Trajectories.

    Returns:
        The spec; the configurations do not depend on the target.
    """
    return pipeline.DataSpec(
        name=f"r30_{scenario}", scenarios=(scenario,), n_per_scenario=n_per,
        criterion=TARGETS[target], cache="disk", window_length=WINDOW_LENGTH,
        horizon=HORIZON, samples_per_trajectory=SAMPLES,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)


def _windowed(runs, spec, seed, input_set, stream):
    return shots.windowed(
        runs, input_set, seed=seed, stream=stream,
        window_length=spec.window_length, horizon=spec.horizon,
        samples_per_trajectory=spec.samples_per_trajectory,
        train_ratio=spec.train_ratio, val_ratio=spec.val_ratio)


def rows_for_seed(
    seed: int,
    n_per: int,
    epochs: int,
    models: Sequence[str],
    input_sets: Sequence[tuple[str, str, int | None]],
    scenarios: Sequence[str] = SCENARIOS,
    targets: Sequence[str] = tuple(TARGETS),
) -> Iterator[dict[str, object]]:
    """Runs every model on every input set for one seed.

    Args:
        seed: Seed of the configurations, the initial states, the split,
            the shot noise and the training.
        n_per: Trajectories per scenario.
        epochs: Training epochs.
        models: Aliases of `model_registry.MODELS`.
        input_sets: (label, kind, n_shots) of the input sets.
        scenarios: Scenario labels.
        targets: Keys of `TARGETS`.

    Yields:
        One row per scenario, target, input set and model that fits it.
    """
    fit = dict(seed=seed, epochs=epochs, batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE)
    for scenario in scenarios:
        for target in targets:
            spec = data_spec(scenario, target, n_per)
            runs = shots.simulate(
                pipeline.system_configs(spec, seed), seed,
                pipeline.STATES_CACHE)
            for label, kind, n_shots in input_sets:
                input_set = shots.InputSet(kind, n_shots)
                train = _windowed(runs, spec, seed, input_set, "train")
                test = _windowed(runs, spec, seed, input_set, "test")
                for alias in models:
                    if alias in TAIL_MODELS and not harness.with_tail(test):
                        continue
                    start = time.perf_counter()
                    model = harness.train_model(alias, train, **fit)
                    yield {
                        "seed": seed, "scenario": scenario,
                        "target": target, "input_set": label,
                        "kind": kind, "n_shots": n_shots or 0,
                        "model": alias, "n_features": len(test.feature_names),
                        "train_seconds": harness.elapsed(start),
                        **harness.score(model, test, spec.horizon)}


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of models, targets and the N grid to the flags."""
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=None)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS),
                        default=list(TARGETS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--with-n-grid", action="store_true",
                        help="also tomography at N = 100 and 10000")


def main() -> None:
    """Runs R30 for the requested seeds.

    Writes experiments/results/r30_finite_shots_2q*.csv.
    """
    args = common.parse_args(__doc__, extra=add_flags)
    n_per, epochs = (30, 3) if args.fast else (500, 100)
    models = args.models or (
        ("physics_lstm", "transformer") if args.fast else MODELS)
    input_sets = INPUT_SETS + (EXTRA_SETS if args.with_n_grid else ())
    path = common.results_path("r30_finite_shots_2q", args)
    config = {"n_per_scenario": n_per, "epochs": epochs,
              "models": list(models), "targets": args.targets,
              "scenarios": args.scenarios,
              "input_sets": [list(s) for s in input_sets],
              "window_length": WINDOW_LENGTH, "horizon": HORIZON,
              "samples_per_trajectory": SAMPLES, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE,
              "split": [TRAIN_RATIO, VAL_RATIO, 1 - TRAIN_RATIO - VAL_RATIO]}
    print(f"R30 | seeds={args.seeds} models={list(models)} -> {path}",
          flush=True)
    with common.ResultsWriter(
            path, COLUMNS, config=config, seeds=args.seeds) as out:
        for seed in args.seeds:
            for row in rows_for_seed(
                    seed, n_per, epochs, models, input_sets,
                    args.scenarios, args.targets):
                out.write(row)
                common.progress("R30", row)


if __name__ == "__main__":
    main()
