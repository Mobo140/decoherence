"""Experiment harness: command line, CSV bytes and provenance sidecar."""
from __future__ import annotations

import csv
import io
import json

import pandas as pd

from experiments import common
from src import pipeline


def test_parse_args_defaults_and_flags():
    args = common.parse_args(argv=[])
    assert (args.seeds, args.fast, args.tag, args.out) == (
        [42, 43, 44], False, "", None)
    args = common.parse_args(
        argv=["--seeds", "1", "2", "--fast", "--tag", "x", "--out", "y.csv"])
    assert (args.seeds, args.fast, args.tag, args.out) == (
        [1, 2], True, "x", "y.csv")


def test_extra_flags_are_added():
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


def test_provenance_sidecar_describes_the_run(tmp_path):
    config = pipeline.ExperimentConfig(
        data=pipeline.DataSpec(name="t", scenarios=("B",)), model="lstm_only",
        tags=(("arm", "x"),))
    path = tmp_path / "out.csv"
    common.write_results([{"a": 1}], path, config=[config], seeds=[7, 8])
    sidecar = tmp_path / "out.csv.provenance.json"
    record = json.loads(sidecar.read_text())
    assert record["csv"] == "out.csv"
    assert record["seeds"] == [7, 8]
    assert record["config"][0]["model"] == "lstm_only"
    assert record["config"][0]["data"]["scenarios"] == ["B"]
    assert {"python", "numpy", "git_commit"} <= set(record["environment"])
    assert path.read_bytes() == b"a\r\n1\r\n"


def test_write_results_takes_a_table_and_default_columns(tmp_path):
    path = tmp_path / "table.csv"
    common.write_results(
        pd.DataFrame({"x": [1, 2], "y": [0.5, 1.5]}), path, config="text")
    assert path.read_bytes() == b"x,y\r\n1,0.5\r\n2,1.5\r\n"
    sidecar = tmp_path / "table.csv.provenance.json"
    assert json.loads(sidecar.read_text())["config"] == "text"
