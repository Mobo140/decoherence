"""Command line: run a recipe file and save a windowed dataset.

    python -m src.blocks.cli generate RECIPE.json --n 100 --seed 0 \\
        --out data/datasets/NAME.npz

The recipe is the JSON of `recipe.recipe_to_dict`. Trajectories are run one
by one, with the states cache in ``data/states``, and a line ``i/N`` is
printed after each so that a job log shows the progress. The dataset is
written with ``np.savez_compressed`` (the persistence of ``src.persistence``
stores trajectories and models, not datasets); see `save_dataset` for the
arrays.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.blocks import runner
from src.blocks.recipe import Recipe, recipe_from_dict, recipe_to_dict
from src.data import dataset as dataset_lib
from src.data.recipe_dataset import to_dataset

DEFAULT_CACHE_DIR = "data/states"
# The dataset's own stream, apart from the sampler (0,) and trajectory
# (1, i) streams of the runner.
_DATASET_STREAM = 2


def _positive_int(text: str) -> int:
    """Parses an integer of at least 1."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an integer.") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}.")
    return value


def _non_negative_int(text: str) -> int:
    """Parses an integer of at least 0."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an integer.") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}.")
    return value


def _ratio(text: str) -> float:
    """Parses a number in [0, 1)."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a number.") from error
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1), got {value}.")
    return value


def _positive_float(text: str) -> float:
    """Parses a positive number."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a number.") from error
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(prog="python -m src.blocks.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser(
        "generate", help="run a recipe and save a windowed dataset")
    gen.add_argument("recipe", type=pathlib.Path, help="recipe JSON file")
    gen.add_argument("--n", type=_positive_int, required=True,
                     help="number of trajectories")
    gen.add_argument("--seed", type=_non_negative_int, default=0)
    gen.add_argument("--out", type=pathlib.Path, required=True,
                     help="dataset file, ending in .npz")
    gen.add_argument("--cache-dir", type=pathlib.Path,
                     default=pathlib.Path(DEFAULT_CACHE_DIR),
                     help="states cache (master-equation recipes)")
    gen.add_argument("--window-length", type=_positive_int, default=20)
    gen.add_argument("--horizon", type=_positive_float, default=1.0)
    gen.add_argument("--samples-per-trajectory", type=_positive_int,
                     default=5)
    gen.add_argument("--train-ratio", type=_ratio, default=0.7)
    gen.add_argument("--val-ratio", type=_ratio, default=0.15)
    gen.add_argument("--min-window-gap", type=_non_negative_int, default=0)
    gen.add_argument("--exclude-censored", dest="include_censored",
                     action="store_false",
                     help="drop trajectories whose target is censored")
    return parser


def load_recipe(path: pathlib.Path) -> Recipe:
    """Reads and validates a recipe file.

    Args:
        path: JSON file written by `recipe.recipe_to_dict`.

    Returns:
        The recipe.

    Raises:
        FileNotFoundError: If there is no such file.
        ValueError: If the file is not JSON or not a valid recipe.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Recipe file {str(path)!r} does not exist.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} is not valid JSON: {error}") from error
    return recipe_from_dict(data)


def _plain(value: Any) -> Any:
    """Turns NumPy scalars into Python ones for `json.dumps`."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable.")


def save_dataset(
    path: pathlib.Path, dataset: dataset_lib.Dataset,
    extra_metadata: dict[str, Any],
) -> None:
    """Writes a dataset to an ``.npz`` file.

    The arrays are the fields of `Dataset` by name; ``feature_names`` is a
    string array, ``metadata`` a JSON string (with `extra_metadata`) and
    each context array is stored as ``context__<name>``.

    Args:
        path: Output file; its parent directory is created.
        dataset: The dataset.
        extra_metadata: Added to the dataset's metadata.

    Raises:
        ValueError: If `path` does not end in ``.npz``.
    """
    if path.suffix != ".npz":
        raise ValueError(f"The output file must end in .npz, got {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        name: getattr(dataset, name) for name in (
            "sequences", "t_obs", "t_decoh_abs", "remaining_time",
            "risk_labels", "train_idx", "val_idx", "test_idx", "censored")}
    arrays["feature_names"] = np.array(dataset.feature_names)
    arrays["metadata"] = np.array(json.dumps(
        {**dataset.metadata, **extra_metadata}, default=_plain))
    for name, values in dataset.context.items():
        arrays[f"context__{name}"] = values
    np.savez_compressed(path, **arrays)


def generate(args: argparse.Namespace) -> pathlib.Path:
    """Runs a recipe and saves the dataset.

    Args:
        args: Parsed arguments of the ``generate`` command.

    Returns:
        The path of the dataset file.

    Raises:
        FileNotFoundError: If the recipe file is missing.
        ValueError: If the recipe or the window options are invalid.
        RuntimeError: If no trajectory yields a window.
    """
    if args.out.suffix != ".npz":
        raise ValueError(f"The output file must end in .npz, got {args.out}.")
    recipe = load_recipe(args.recipe)
    cache_dir = (None if recipe.evolve.fn == "evolve.sequential_record"
                 else args.cache_dir)
    if recipe.seeds == "spawn":
        # A spawned trajectory depends on (seed, i) only, so they can be
        # run one at a time to report the progress.
        runs = []
        for i in range(args.n):
            runs += runner.run(
                recipe, args.n, args.seed, cache_dir=cache_dir, indices=[i])
            print(f"{i + 1}/{args.n}", flush=True)
    else:
        # Legacy trajectories depend on their predecessors.
        runs = runner.run(recipe, args.n, args.seed, cache_dir=cache_dir)
        print(f"{args.n}/{args.n}", flush=True)
    print("building the dataset", flush=True)
    dataset = to_dataset(
        runs,
        window_length=args.window_length, horizon=args.horizon,
        samples_per_trajectory=args.samples_per_trajectory,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        min_window_gap=args.min_window_gap,
        include_censored=args.include_censored,
        rng=np.random.default_rng(np.random.SeedSequence(
            args.seed, spawn_key=(_DATASET_STREAM,))))
    save_dataset(args.out, dataset, {
        "recipe": recipe_to_dict(recipe), "run_seed": args.seed,
        "n_recipe_trajectories": args.n})
    print(f"saved {args.out}: {len(dataset.sequences)} windows, "
          f"{len(dataset.feature_names)} features", flush=True)
    return args.out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments after the program name; None reads ``sys.argv``.

    Returns:
        0 on success, 2 for a bad argument, a missing or invalid recipe, or
        a dataset that cannot be built (the message is on stderr).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        generate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
