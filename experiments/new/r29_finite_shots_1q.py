"""R29 — one qubit with inputs estimated from a finite number of shots.

Why
---
The models of the paper read exact ensemble means of rho(t). An experiment
gives 0/1 outcomes, and each mean comes from N repeated preparations. For
one qubit the five inputs sigma_x, sigma_y, sigma_z, purity and
coherence_l1 follow from the three Pauli means by linear inversion, so
`local` and `tomography` settings are the same measurement (3 settings).
The shot noise of a mean m is sqrt((1 - m^2) / N), about 0.03 at m = 0 for
N = 1000, which R15 (Gaussian noise, same windows) can be read against.

Scenarios B and C, 100 trajectories each, window 50 (the R15 data; the
exact inputs reproduce its windows bit for bit). Two arms per model:

  train_exact  trained on exact inputs, scored on N-shot inputs
  train_shots  trained on N-shot inputs, scored on N-shot inputs of the same
               N (noise-aware training, as R16); the train and test windows
               have independent shot noise (streams "train" and "test")

N in {100, 300, 1000, 3000, 10000} and the exact inputs (n_shots = 0 in the
CSV). The windows, the split and the test trajectories are the same for all
N. Rows are scored on the pooled test windows ("B+C") and on those of each
scenario.

Writes ONLY to experiments/results/r29_finite_shots_1q*.csv with its
provenance file.

Usage
-----
    python -m experiments.new.r29_finite_shots_1q --fast --tag smoke
    python -m experiments.new.r29_finite_shots_1q --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Iterator, Sequence

from experiments import common
from experiments.new import shot_harness as harness
from src import pipeline
from src.data import shots

N_SHOTS = (100, 300, 1000, 3000, 10000)
MODELS = ("physics_only", "physics_lstm", "lstm_only", "transformer")
ARMS = ("train_exact", "train_shots")
COLUMNS = ("seed", "arm", "model", "n_shots", "scenario", "r2", "mae",
           "auroc", "n_samples", "train_seconds")
# R15 windows and the pipeline's training defaults.
WINDOW_LENGTH = 50
HORIZON = 1.0
SAMPLES = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
# dissipator_type code of scenario B (sigma_minus) and C (sigma_z).
SCENARIO_CODES = {"B": 0.0, "C": 1.0}


def data_spec(n_per: int) -> pipeline.DataSpec:
    """Declares the R15 data: scenarios B and C.

    Args:
        n_per: Trajectories per scenario.

    Returns:
        The spec; its split is 0.8 / 0.1 / 0.1 as in R15.
    """
    return pipeline.DataSpec(
        name="r29", scenarios=("B", "C"), n_per_scenario=n_per,
        cache="disk", window_length=WINDOW_LENGTH, horizon=HORIZON,
        samples_per_trajectory=SAMPLES)


def _windowed(runs, spec, seed, input_set, stream):
    return shots.windowed(
        runs, input_set, seed=seed, stream=stream,
        window_length=spec.window_length, horizon=spec.horizon,
        samples_per_trajectory=spec.samples_per_trajectory,
        train_ratio=spec.train_ratio, val_ratio=spec.val_ratio)


def _scored_rows(
    model, dataset, spec, base: dict
) -> Iterator[dict[str, object]]:
    """Scores the test windows pooled and per scenario."""
    codes = dataset.context["dissipator_type"]
    yield {**base, "scenario": "B+C",
           **harness.score(model, dataset, spec.horizon)}
    for label, code in SCENARIO_CODES.items():
        part = harness.restrict_test(dataset, codes == code)
        yield {**base, "scenario": label,
               **harness.score(model, part, spec.horizon)}


def rows_for_seed(
    seed: int,
    n_per: int,
    epochs: int,
    models: Sequence[str],
    n_shots: Sequence[int],
) -> Iterator[dict[str, object]]:
    """Runs both arms of every model for one seed.

    Args:
        seed: Seed of the configurations, the initial states, the split,
            the shot noise and the training.
        n_per: Trajectories per scenario.
        epochs: Training epochs.
        models: Aliases of `model_registry.MODELS`.
        n_shots: Shot counts.

    Yields:
        One row per arm, model, N and scenario; N = 0 is the exact input.
    """
    spec = data_spec(n_per)
    runs = shots.simulate(
        pipeline.system_configs(spec, seed), seed, pipeline.STATES_CACHE)
    exact_set = shots.InputSet("exact")
    exact = _windowed(runs, spec, seed, exact_set, "test")
    shot_sets = {n: shots.InputSet("local", n) for n in n_shots}
    test = {n: _windowed(runs, spec, seed, s, "test")
            for n, s in shot_sets.items()}
    train = {n: _windowed(runs, spec, seed, s, "train")
             for n, s in shot_sets.items()}
    fit = dict(seed=seed, epochs=epochs, batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE)
    for alias in models:
        start = time.perf_counter()
        model = harness.train_model(alias, exact, **fit)
        seconds = harness.elapsed(start)
        for n, dataset in [(0, exact), *test.items()]:
            base = dict(seed=seed, arm="train_exact", model=alias,
                        n_shots=n, train_seconds=seconds)
            yield from _scored_rows(model, dataset, spec, base)
        for n in n_shots:
            start = time.perf_counter()
            model = harness.train_model(alias, train[n], **fit)
            base = dict(seed=seed, arm="train_shots", model=alias,
                        n_shots=n, train_seconds=harness.elapsed(start))
            yield from _scored_rows(model, test[n], spec, base)


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of models and shot counts to the common flags."""
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        default=None)
    parser.add_argument("--n-shots", type=int, nargs="+", default=None)


def main() -> None:
    """Runs R29 for the requested seeds.

    Writes experiments/results/r29_finite_shots_1q*.csv.
    """
    args = common.parse_args(__doc__, extra=add_flags)
    n_per, epochs = (30, 3) if args.fast else (100, 100)
    models = args.models or (
        ("physics_lstm", "transformer") if args.fast else MODELS)
    n_shots = args.n_shots or ((100, 1000) if args.fast else N_SHOTS)
    path = common.results_path("r29_finite_shots_1q", args)
    config = {"n_per_scenario": n_per, "epochs": epochs,
              "models": list(models), "n_shots": list(n_shots),
              "window_length": WINDOW_LENGTH, "horizon": HORIZON,
              "samples_per_trajectory": SAMPLES, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "split": [0.8, 0.1, 0.1]}
    print(f"R29 | seeds={args.seeds} models={list(models)} "
          f"N={list(n_shots)} -> {path}", flush=True)
    with common.ResultsWriter(
            path, COLUMNS, config=config, seeds=args.seeds) as out:
        for seed in args.seeds:
            for row in rows_for_seed(seed, n_per, epochs, models, n_shots):
                out.write(row)
                if row["scenario"] == "B+C":
                    common.progress("R29", row)


if __name__ == "__main__":
    main()
