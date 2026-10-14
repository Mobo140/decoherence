"""Shared harness of the experiment scripts.

A script declares its arms (`src.pipeline.ExperimentConfig`), parses the
common command line, and writes the rows of `pipeline.iter_rows` to a CSV
with `ResultsWriter`. Next to every CSV a provenance file
`<csv>.provenance.json` records the configuration, the seeds and the
software environment; it is not part of the CSV.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import pathlib
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from src import pipeline
from src import reproducibility

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
DEFAULT_SEEDS = (42, 43, 44)


def parse_args(
    description: str | None = None,
    *,
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: Sequence[str] | None = None,
    seeds: Sequence[int] | None = DEFAULT_SEEDS,
    out: bool = True,
) -> argparse.Namespace:
    """Parses the command line shared by the experiments.

    Flags: `--seeds` (integers), `--fast` (smoke run on reduced data),
    `--tag` (suffix of the output name) and `--out` (output file name
    under `experiments/results/`, replacing the default name).

    Args:
        description: Text of the help message.
        extra: Adds the script's own flags to the parser.
        argv: Arguments to parse; defaults to `sys.argv[1:]`.
        seeds: Default of `--seeds`; `None` leaves the flag out, for
            experiments that are a single run with a fixed seed.
        out: Whether to offer `--out`; `False` for scripts that write
            several CSVs.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    if seeds is not None:
        parser.add_argument("--seeds", type=int, nargs="+",
                            default=list(seeds))
    parser.add_argument("--fast", action="store_true",
                        help="smoke run on reduced data")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix of the output file name")
    if out:
        parser.add_argument("--out", type=str, default=None,
                            help="output CSV name under experiments/results/")
    if extra is not None:
        extra(parser)
    return parser.parse_args(argv)


def parse_fast(
    description: str | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parses the only flag of the runners that chain experiments.

    Args:
        description: Text of the help message.
        argv: Arguments to parse; defaults to `sys.argv[1:]`.

    Returns:
        The parsed arguments; `fast` is passed on to every experiment.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true",
                        help="smoke run of every experiment on reduced data")
    return parser.parse_args(argv)


def results_path(stem: str, args: argparse.Namespace) -> pathlib.Path:
    """Returns where a script writes its CSV.

    Args:
        stem: File name without suffix, e.g. "r22_ar1_noise_training".
        args: Arguments from `parse_args`.

    Returns:
        `--out` under the results directory if given, otherwise
        `<stem>[_<tag>][_fast].csv` there.
    """
    out = getattr(args, "out", None)
    if out:
        return RESULTS_DIR / out
    suffix = (f"_{args.tag}" if args.tag else "") + (
        "_fast" if args.fast else "")
    return RESULTS_DIR / f"{stem}{suffix}.csv"


def _jsonable(value: Any) -> Any:
    """Converts configurations to something `json` can write."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return value


def write_provenance(
    csv_path: pathlib.Path,
    *,
    config: object = None,
    seeds: Iterable[int] | None = None,
) -> pathlib.Path:
    """Writes `<csv>.provenance.json` next to a results file.

    Args:
        csv_path: The results file.
        config: What produced it: an `ExperimentConfig`, several, a mapping
            or a description text.
        seeds: Seeds of the run.

    Returns:
        The path of the provenance file.
    """
    path = csv_path.with_name(csv_path.name + ".provenance.json")
    record = {
        "csv": csv_path.name,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "command": [pathlib.Path(sys.argv[0]).name, *sys.argv[1:]],
        "seeds": None if seeds is None else list(seeds),
        "config": _jsonable(config),
        "environment": reproducibility.environment_info(),
    }
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


class ResultsWriter:
    """Writes result rows to a CSV as they arrive.

    The file is flushed after every row, so an interrupted run keeps what
    it computed. Entering the context writes the header and the provenance
    file.

    Attributes:
        path: The CSV file.
        columns: Column names in file order. Keys of a row that are not
            columns are ignored; a missing key gives an empty cell.
    """

    def __init__(
        self,
        path: pathlib.Path,
        columns: Sequence[str],
        *,
        config: object = None,
        seeds: Iterable[int] | None = None,
    ) -> None:
        """Prepares the writer; nothing is written yet.

        Args:
            path: The CSV file; overwritten.
            columns: Column names in file order.
            config: Recorded in the provenance file (see
                `write_provenance`).
            seeds: Recorded in the provenance file.
        """
        self.path = path
        self.columns = list(columns)
        self._config = config
        self._seeds = None if seeds is None else list(seeds)
        self._file = None
        self._writer = None

    def __enter__(self) -> ResultsWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        write_provenance(self.path, config=self._config, seeds=self._seeds)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    def write(self, row: Mapping[str, object]) -> None:
        """Appends one row and flushes it to disk."""
        self._writer.writerow(row)
        self._file.flush()


def write_results(
    rows: Iterable[Mapping[str, object]] | pd.DataFrame,
    path: pathlib.Path,
    *,
    columns: Sequence[str] | None = None,
    config: object = None,
    seeds: Iterable[int] | None = None,
) -> None:
    """Writes all rows to a CSV and its provenance file.

    Args:
        rows: Row dicts, or a table (whose missing cells are then written
            as nan).
        path: The CSV file; overwritten.
        columns: Column names in file order; defaults to the keys of the
            first row.
        config: Recorded in the provenance file (see `write_provenance`).
        seeds: Recorded in the provenance file.
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    rows = list(rows)
    if columns is None:
        columns = list(rows[0])
    with ResultsWriter(path, columns, config=config, seeds=seeds) as out:
        for row in rows:
            out.write(row)


def _format(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def progress(label: str, row: Mapping[str, object]) -> None:
    """Prints one result row on a line."""
    fields = " ".join(
        f"{key}={_format(value)}" for key, value in row.items()
        if value != "" and key not in ("n_train_windows", "n_censored"))
    print(f"{label} | {fields}", flush=True)


def run_arms(
    label: str,
    configs: Sequence[pipeline.ExperimentConfig],
    args: argparse.Namespace,
    path: pathlib.Path,
    columns: Sequence[str],
    rename: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Runs the arms for the requested seeds and writes the CSV.

    Args:
        label: Experiment label for the progress lines, e.g. "R22".
        configs: The arms in the order of the result rows.
        args: Arguments from `parse_args`; `seeds` is used.
        path: The CSV file.
        columns: Column names in file order (after the renaming).
        rename: Pipeline column -> name in the CSV, for results files whose
            columns predate the pipeline's names.

    Returns:
        All rows, renamed, with every key the pipeline produced (not only
        the columns written), for a summary printed after the run.
    """
    print(f"{label} | seeds={args.seeds} arms={len(configs)} -> {path}",
          flush=True)
    rows = []
    with ResultsWriter(
            path, columns, config=configs, seeds=args.seeds) as out:
        for row in pipeline.iter_rows(configs, args.seeds):
            if rename:
                row = {rename.get(key, key): value
                       for key, value in row.items()}
            out.write(row)
            progress(label, row)
            rows.append(row)
    return rows
