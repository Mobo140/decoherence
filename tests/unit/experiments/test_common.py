"""experiments.common: command line, results path and CSV bytes."""
from __future__ import annotations

import csv
import io
import json

import pandas as pd

from experiments import common


def test_parse_args_gives_defaults_and_reads_flags():
    args = common.parse_args(argv=[])
    assert (args.seeds, args.fast, args.tag, args.out) == (
        [42, 43, 44], False, "", None)
    args = common.parse_args(
        argv=["--seeds", "1", "2", "--fast", "--tag", "x", "--out", "y.csv"])
    assert (args.seeds, args.fast, args.tag, args.out) == (
        [1, 2], True, "x", "y.csv")


def test_parse_args_seeds_none_drops_the_flag():
    args = common.parse_args(argv=["--fast"], seeds=None)
    assert not hasattr(args, "seeds") and args.fast
    args = common.parse_args(argv=[], seeds=(7,))
    assert args.seeds == [7]


def test_parse_args_adds_extra_flags():
    args = common.parse_args(
        extra=lambda p: p.add_argument("--part", default="a"),
        argv=["--part", "b"])
    assert args.part == "b"


def test_results_path_adds_tag_and_fast_and_honours_out():
    path = common.results_path("stem", common.parse_args(argv=[]))
    assert path == common.RESULTS_DIR / "stem.csv"
    path = common.results_path(
        "stem", common.parse_args(argv=["--tag", "t", "--fast"]))
    assert path.name == "stem_t_fast.csv"
    path = common.results_path(
        "stem", common.parse_args(argv=["--out", "own.csv", "--fast"]))
    assert path == common.RESULTS_DIR / "own.csv"


def _expected_bytes(rows, columns):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def test_csv_bytes_equal_a_plain_dict_writer(tmp_path):
    rows = [{"seed": 1, "arm": "a", "r2": 0.1 + 0.2, "auroc": float("nan"),
             "s": ""},
            {"seed": 2, "arm": "b", "r2": 1e-7, "extra": 5}]
    columns = ["seed", "arm", "s", "r2", "auroc"]
    path = tmp_path / "out.csv"
    with common.ResultsWriter(path, columns) as writer:
        for row in rows:
            writer.write(row)
    plain_rows = [{k: v for k, v in r.items() if k in columns} for r in rows]
    assert path.read_bytes() == _expected_bytes(
        plain_rows, columns).encode()


def test_write_results_takes_a_table_and_default_columns(tmp_path):
    path = tmp_path / "table.csv"
    common.write_results(
        pd.DataFrame({"x": [1, 2], "y": [0.5, 1.5]}), path, config="text")
    assert path.read_bytes() == b"x,y\r\n1,0.5\r\n2,1.5\r\n"
    sidecar = tmp_path / "table.csv.provenance.json"
    assert json.loads(sidecar.read_text())["config"] == "text"
